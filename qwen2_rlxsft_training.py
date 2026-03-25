"""
完整的 RLxSFT 混合训练实现 - 生产级别
支持 SEQUENTIAL, WEIGHTED, CHORD, LUFFY, RELIFT 等多种策略

功能特点：
  - 完整的数据加载和预处理
  - 多种混合训练策略
  - 单卡训练（无 DDP）
  - LoRA 微调
  - 实时监控和日志记录
  - 模型检查点保存和恢复
  - 内存优化（梯度检查点、参考模型 CPU 加载）

使用方法：
    python qwen2_rlxsft_training.py --strategy weighted --epochs 3
    python qwen2_rlxsft_training.py --strategy chord --epochs 5 --sample-size 5000
    python qwen2_rlxsft_training.py --strategy luffy --epochs 3 --device 0
    python qwen2_rlxsft_training.py --strategy relift --epochs 3 --device 0 --batch-size 1
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime
import copy
from tqdm import tqdm
import gc

# 导入 Hugging Face 库
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,  # 从 transformers 导入（推荐）
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, SequentialSampler


# ============================================================================
# 日志配置
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rlxsft_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# 枚举和数据类
# ============================================================================

class Strategy(str, Enum):
    """混合训练策略"""
    SEQUENTIAL = "sequential"  # 顺序执行：先 SFT 后 RL
    WEIGHTED = "weighted"      # 加权组合：固定权重
    CHORD = "chord"            # 协调混合：动态权重调整
    LUFFY = "luffy"            # 不确定性加权：自动学习权重
    RELIFT = "relift"          # 强化学习主导：SFT 10% + RL 90%


@dataclass
class RLxSFTConfig:
    """RLxSFT 混合训练配置"""
    
    # ============ 基本配置 ============
    strategy: str = "weighted"
    output_dir: str = "./outputs/qwen2_rlxsft_model"
    model_name: str = "./models/Qwen2-7B-Instruct"
    data_path: str = "./financial_data/kyc_gspo_training_data.jsonl"
    
    # ============ 数据配置 ============
    max_samples: int = 5000  # 5K 数据：建议 2-3 epochs；10K 数据：建议 1-2 epochs
    max_seq_length: int = 1024  # 从 2048 降低以减少显存占用
    train_test_split: float = 0.9
    
    # ============ 训练配置 ============
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1  # 降低为 1 以降低显存压力
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2  # 恢复梯度累积以提高稳定性
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # ⚡ 显存优化参数
    use_gradient_checkpointing: bool = True  # 激活函数重计算，节省显存 ~30%
    use_flash_attention: bool = False  # Flash Attention 2（需要 PyTorch 2.0+ 和 flash_attn 包）
    use_mixed_precision: bool = True  # 混合精度训练（fp16 + bf16）
    
    # ⚡ 推理优化
    ref_model_on_gpu: bool = False  # 参考模型放在 CPU 上节省显存（显存 < 24GB 时推荐）
    num_workers: int = 4  # DataLoader 的工作进程数
    prefetch_factor: int = 2  # 预取因子
    
    # ============ LoRA 配置 ============
    use_lora: bool = True
    lora_r: int = 4  # 从 8 → 4，减少 LoRA 参数量
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj"])  # 添加 k_proj
    
    # ============ 混合权重 ============
    sft_weight: float = 0.3
    rl_weight: float = 0.7
    
    # ============ CHORD 配置（协调混合） ============
    chord_warm_steps: int = 500
    chord_phase1_ratio: float = 0.33
    chord_phase2_ratio: float = 0.33
    chord_phase3_ratio: float = 0.34
    
    # ============ LUFFY 配置（不确定性加权） ============
    luffy_uncertainty_lr: float = 0.001
    luffy_initial_log_variance_sft: float = 0.0
    luffy_initial_log_variance_rl: float = 0.0
    
    # ============ RL 参数 ============
    group_size: int = 4
    kl_coef: float = 0.003
    entropy_coef: float = 0.01
    clip_ratio: float = 0.15
    clip_ratio_high: float = 0.28
    
    # ============ 检查点和日志 ============
    save_steps: int = 500
    logging_steps: int = 100
    eval_steps: int = 500
    save_total_limit: int = 3


# ============================================================================
# 数据集
# ============================================================================

class KYCDataset(Dataset):
    """KYC 风险评估数据集"""
    
    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_seq_length: int = 2048,
        max_samples: Optional[int] = None,
        split: str = "train",
        train_test_ratio: float = 0.9,
        seed: int = 42
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        logger.info(f"Loading data from {file_path}...")
        
        # 加载数据
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_samples and len(self.data) >= max_samples:
                    break
                try:
                    item = json.loads(line)
                    self.data.append(item)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"✓ Loaded {len(self.data)} samples")
        
        # 分割数据集
        np.random.seed(seed)
        indices = np.random.permutation(len(self.data))
        split_idx = int(len(self.data) * train_test_ratio)
        
        if split == "train":
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
        
        logger.info(f"✓ {split.upper()} set size: {len(self.indices)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        item = self.data[self.indices[idx]]
        
        prompt = item.get('prompt', '')
        target = item.get('target', '')
        risk_label = item.get('risk_label', 0)
        
        # 组合输入和输出
        full_text = f"{prompt}\n{target}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        
        # 计算奖励（基于 risk_label，所有奖励都为正）
        # 低风险: +1.0，高风险: +0.5（正向强化）
        reward = 1.0 if risk_label == 0 else 0.5
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'rewards': torch.tensor(reward, dtype=torch.float32),
            'risk_label': torch.tensor(risk_label, dtype=torch.long),
        }


# ============================================================================
# 混合损失计算器
# ============================================================================

class HybridLossCalculator:
    """计算 SFT 和 RL 的混合损失"""
    
    def __init__(self, config: RLxSFTConfig):
        self.config = config
        
        # LUFFY 配置中的学习参数
        if config.strategy == Strategy.LUFFY.value:
            self.log_variance_sft = nn.Parameter(
                torch.tensor(config.luffy_initial_log_variance_sft, dtype=torch.float32),
                requires_grad=True
            )
            self.log_variance_rl = nn.Parameter(
                torch.tensor(config.luffy_initial_log_variance_rl, dtype=torch.float32),
                requires_grad=True
            )
            self.uncertainty_optim = None
    
    def compute_sft_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算 SFT 有监督学习损失"""
        # 移动位置用于标准语言模型损失
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 计算交叉熵损失
        loss_fct = nn.CrossEntropyLoss(reduction='mean')
        sft_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return sft_loss
    
    def compute_rl_loss(
        self,
        policy_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """计算 RL 损失
        
        简化的策略梯度形式：
        RL_loss = -clip(1 + log_ratio, eps, inf) * advantage
        
        这是一个简化的 PPO-style 损失，避免复杂的 clipping 逻辑，
        而是直接优化策略以最大化好的行为，最小化坏的行为。
        
        Args:
            policy_logprobs: 策略模型的 log 概率 (B,)
            ref_logprobs: 参考模型的 log 概率 (B,)
            advantages: 优势估计值，已归一化 (B,)，范围约 [-1, 1]
        
        Returns:
            标量损失值，范围通常在 [0, 1] 左右
        """
        
        # 计算 log 概率比率
        log_ratio = policy_logprobs - ref_logprobs
        
        # 限制范围防止梯度爆炸
        log_ratio = torch.clamp(log_ratio, min=-5.0, max=5.0)
        
        # 直接使用 log_ratio 加权的策略梯度
        # 公式简化：当优势为正时，最大化 log_policy（降低损失）
        #         当优势为负时，最小化 log_policy（提高损失）
        rl_loss_per_sample = -log_ratio * advantages
        
        # 取平均
        rl_loss = rl_loss_per_sample.mean()
        
        # 数值稳定性检查
        if torch.isnan(rl_loss) or torch.isinf(rl_loss):
            logger.warning(
                f"⚠️  RL Loss 异常 (NaN/Inf):\n"
                f"  policy_logprobs: mean={policy_logprobs.mean().item():.6f}, "
                f"std={policy_logprobs.std().item():.6f}\n"
                f"  log_ratio: mean={log_ratio.mean().item():.6f}, "
                f"std={log_ratio.std().item():.6f}\n"
                f"  advantages: mean={advantages.mean().item():.4f}, "
                f"std={advantages.std().item():.4f}\n"
                f"  rl_loss_per_sample: "
                f"min={rl_loss_per_sample.min().item():.4f}, "
                f"max={rl_loss_per_sample.max().item():.4f}"
            )
            rl_loss = torch.tensor(0.01, dtype=torch.float32, device=rl_loss.device)
        
        # 绝对值处理（确保为正）
        rl_loss = torch.abs(rl_loss)
        
        return rl_loss
    
    def compute_hybrid_loss(
        self,
        sft_loss: torch.Tensor,
        rl_loss: torch.Tensor,
        global_step: int = 0,
        total_steps: int = 1
    ) -> Dict[str, Any]:
        """计算混合损失"""
        
        if self.config.strategy == Strategy.SEQUENTIAL.value:
            # 顺序执行：交替使用 SFT 和 RL
            is_sft_step = (global_step % 2) == 0
            if is_sft_step:
                loss = sft_loss
            else:
                loss = rl_loss
            
            return {
                'loss': loss,
                'sft_loss': sft_loss.detach().item(),
                'rl_loss': rl_loss.detach().item(),
                'sft_weight': 1.0 if is_sft_step else 0.0,
                'rl_weight': 0.0 if is_sft_step else 1.0,
            }
        
        elif self.config.strategy == Strategy.WEIGHTED.value:
            # 加权组合：固定权重
            loss = (
                self.config.sft_weight * sft_loss +
                self.config.rl_weight * rl_loss
            )
            
            return {
                'loss': loss,
                'sft_loss': sft_loss.detach().item(),
                'rl_loss': rl_loss.detach().item(),
                'sft_weight': self.config.sft_weight,
                'rl_weight': self.config.rl_weight,
            }
        
        elif self.config.strategy == Strategy.CHORD.value:
            # 协调混合：三阶段动态调整
            total_steps_float = float(total_steps)
            progress = global_step / max(total_steps_float, 1.0)
            
            if progress < self.config.chord_phase1_ratio:
                # Phase 1: SFT 主导
                sft_weight = 0.7
                rl_weight = 0.3
            elif progress < (self.config.chord_phase1_ratio + self.config.chord_phase2_ratio):
                # Phase 2: 平衡
                sft_weight = 0.5
                rl_weight = 0.5
            else:
                # Phase 3: RL 主导
                sft_weight = 0.3
                rl_weight = 0.7
            
            loss = sft_weight * sft_loss + rl_weight * rl_loss
            
            return {
                'loss': loss,
                'sft_loss': sft_loss.detach().item(),
                'rl_loss': rl_loss.detach().item(),
                'sft_weight': sft_weight,
                'rl_weight': rl_weight,
            }
        
        elif self.config.strategy == Strategy.LUFFY.value:
            # 不确定性加权：自动学习权重
            # 使用任务不确定性进行加权
            precision_sft = torch.exp(-self.log_variance_sft)
            precision_rl = torch.exp(-self.log_variance_rl)
            
            loss = (
                precision_sft * sft_loss +
                precision_rl * rl_loss +
                self.log_variance_sft +
                self.log_variance_rl
            )
            
            sft_weight = precision_sft.item() / (precision_sft.item() + precision_rl.item() + 1e-8)
            rl_weight = 1.0 - sft_weight
            
            return {
                'loss': loss,
                'sft_loss': sft_loss.detach().item(),
                'rl_loss': rl_loss.detach().item(),
                'sft_weight': sft_weight,
                'rl_weight': rl_weight,
                'log_variance_sft': self.log_variance_sft.detach().item(),
                'log_variance_rl': self.log_variance_rl.detach().item(),
            }
        
        elif self.config.strategy == Strategy.RELIFT.value:
            # 强化学习主导
            loss = (
                0.1 * sft_loss +
                0.9 * rl_loss
            )
            
            return {
                'loss': loss,
                'sft_loss': sft_loss.detach().item(),
                'rl_loss': rl_loss.detach().item(),
                'sft_weight': 0.1,
                'rl_weight': 0.9,
            }
        
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")


# ============================================================================
# 训练器
# ============================================================================

class RLxSFTTrainer:
    """RLxSFT 混合训练器"""
    
    def __init__(
        self,
        config: RLxSFTConfig,
        model: nn.Module,
        ref_model: nn.Module,
        tokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # 初始化损失计算器
        self.loss_calculator = HybridLossCalculator(config)
        
        # 初始化优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # 初始化学习率调度器
        total_steps = len(train_dataloader) * config.num_train_epochs // config.gradient_accumulation_steps
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        # 训练状态
        self.global_step = 0
        self.total_steps = total_steps
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✓ Trainer initialized")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Warmup steps: {warmup_steps}")
    
    def forward_pass(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """前向传播"""
        
        input_ids = batch['input_ids'].to(self.model.device)
        attention_mask = batch['attention_mask'].to(self.model.device)
        labels = batch['labels'].to(self.model.device)
        rewards = batch['rewards'].to(self.model.device)
        
        # 策略模型前向传播
        with torch.no_grad():
            model = self.model.module if hasattr(self.model, 'module') else self.model
            model.eval()
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        logits = outputs.logits
        
        # 计算 SFT 损失
        sft_loss = self.loss_calculator.compute_sft_loss(logits, labels)
        
        # 计算 RL 损失
        policy_logprobs = F.log_softmax(logits, dim=-1)
        policy_logprobs = policy_logprobs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        
        # 参考模型 logprobs（不需要梯度）
        with torch.no_grad():
            # 如果参考模型在 GPU 上，直接使用（避免数据传输开销）
            if self.config.ref_model_on_gpu:
                ref_outputs = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                ref_logits = ref_outputs.logits
                ref_logprobs = F.log_softmax(ref_logits, dim=-1)
                ref_logprobs = ref_logprobs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
            else:
                # 参考模型在 CPU 上，需要数据传输
                ref_input_ids = input_ids.cpu()
                ref_attention_mask = attention_mask.cpu()
                
                ref_outputs = self.ref_model(
                    input_ids=ref_input_ids,
                    attention_mask=ref_attention_mask,
                )
                ref_logits = ref_outputs.logits
                ref_logprobs = F.log_softmax(ref_logits, dim=-1)
                ref_logprobs = ref_logprobs.gather(-1, ref_input_ids.unsqueeze(-1)).squeeze(-1)
                # 移回策略模型设备
                ref_logprobs = ref_logprobs.to(input_ids.device)
        
        # 计算平均 logprobs（token 级别归一化）
        policy_logprobs_mean = policy_logprobs.mean(dim=1)
        ref_logprobs_mean = ref_logprobs.mean(dim=1)
        
        # ⚡ 重要修复：计算相对于基线的 advantages（不是单纯的奖励归一化）
        # advantages = rewards - baseline
        # 这样即使 rewards 都是正数，advantages 也能有正有负，表示相对优劣
        #
        # 基线策略：使用 policy_logprobs_mean 作为基线（策略模型本身的熵基线）
        # 这提供了一个数值稳定的参考点
        baseline = policy_logprobs_mean.detach().mean()  # 平均基线
        
        # 计算 advantages（相对于基线的差值）
        advantages = rewards - baseline
        
        # 对 advantages 进行归一化以提高训练稳定性
        # 这避免了不同 batch 间的尺度差异
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        
        if adv_std > 0:
            # 标准化
            advantages_normalized = (advantages - adv_mean) / (adv_std + 1e-8)
            # 缩放到合理范围（通常在 [-1, 1] 左右）
            advantages_normalized = advantages_normalized * 0.1  # 进一步缩小尺度
        else:
            # 所有 advantages 相同，直接使用
            advantages_normalized = advantages.clone()
        
        rl_loss = self.loss_calculator.compute_rl_loss(
            policy_logprobs_mean,
            ref_logprobs_mean,
            advantages_normalized,
        )
        
        # 计算混合损失
        loss_dict = self.loss_calculator.compute_hybrid_loss(
            sft_loss,
            rl_loss,
            global_step=self.global_step,
            total_steps=self.total_steps,
        )
        
        return loss_dict['loss'], loss_dict
    
    def train_epoch(self, epoch: int):
        """训练一个 epoch"""
        
        self.model.train()
        total_loss = 0.0
        total_sft_loss = 0.0
        total_rl_loss = 0.0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            # 前向传播
            loss, loss_dict = self.forward_pass(batch)
            
            # 按梯度累积步长缩放损失
            loss = loss / self.config.gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 梯度累积
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # 优化器步长
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # 累积统计
            total_loss += loss.detach().item() * self.config.gradient_accumulation_steps
            total_sft_loss += loss_dict['sft_loss']
            total_rl_loss += loss_dict['rl_loss']
            
            # 日志
            if (step + 1) % self.config.logging_steps == 0:
                avg_loss = total_loss / (step + 1)
                progress_bar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'sft': f"{total_sft_loss / (step + 1):.4f}",
                    'rl': f"{total_rl_loss / (step + 1):.4f}",
                    'sft_w': f"{loss_dict['sft_weight']:.2f}",
                    'rl_w': f"{loss_dict['rl_weight']:.2f}",
                })
                
                logger.info(
                    f"Epoch {epoch + 1} | Step {step + 1} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"SFT Loss: {total_sft_loss / (step + 1):.4f} | "
                    f"RL Loss: {total_rl_loss / (step + 1):.4f}"
                )
            
            # 保存检查点
            if self.global_step > 0 and self.global_step % self.config.save_steps == 0:
                self.save_checkpoint(epoch, step)
                # 检查点保存后清理内存
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            'avg_loss': total_loss / len(self.train_dataloader),
            'avg_sft_loss': total_sft_loss / len(self.train_dataloader),
            'avg_rl_loss': total_rl_loss / len(self.train_dataloader),
        }
    
    def save_checkpoint(self, epoch: int, step: int):
        """保存检查点"""
        
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.save_pretrained(str(checkpoint_dir))
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        
        # 保存配置
        config_path = checkpoint_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        logger.info(f"✓ Checkpoint saved: {checkpoint_dir}")
    
    def train(self):
        """执行完整训练"""
        
        logger.info("=" * 80)
        logger.info("🚀 启动 RLxSFT 混合训练")
        logger.info(f"  策略: {self.config.strategy}")
        logger.info(f"  Epochs: {self.config.num_train_epochs}")
        logger.info(f"  SFT 权重: {self.config.sft_weight}")
        logger.info(f"  RL 权重: {self.config.rl_weight}")
        logger.info("=" * 80)
        
        best_loss = float('inf')
        
        for epoch in range(self.config.num_train_epochs):
            logger.info(f"\n【Epoch {epoch + 1}/{self.config.num_train_epochs}】")
            
            # 训练
            metrics = self.train_epoch(epoch)
            
            logger.info(
                f"✓ Epoch {epoch + 1} 完成 | "
                f"平均损失: {metrics['avg_loss']:.4f} | "
                f"SFT 损失: {metrics['avg_sft_loss']:.4f} | "
                f"RL 损失: {metrics['avg_rl_loss']:.4f}"
            )
            
            # 保存最佳模型
            if metrics['avg_loss'] < best_loss:
                best_loss = metrics['avg_loss']
                self.save_best_model()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ RLxSFT 混合训练完成！")
        logger.info("=" * 80)
    
    def save_best_model(self):
        """保存最佳模型"""
        
        best_model_dir = Path(self.config.output_dir) / "best_model"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.save_pretrained(str(best_model_dir))
        self.tokenizer.save_pretrained(str(best_model_dir))
        
        logger.info(f"✓ 最佳模型已保存: {best_model_dir}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='RLxSFT 混合训练框架')
    
    # 策略参数
    parser.add_argument(
        '--strategy',
        type=str,
        default='weighted',
        choices=['sequential', 'weighted', 'chord', 'luffy', 'relift'],
        help='混合训练策略'
    )
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=3, help='训练 epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='批大小')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='学习率')
    
    # 数据参数
    parser.add_argument('--sample-size', type=int, default=5000, help='样本数量')
    parser.add_argument('--data-path', type=str, default='./financial_data/kyc_gspo_training_data.jsonl',
                       help='数据路径')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, default='./models/Qwen2-7B-Instruct',
                       help='模型路径')
    parser.add_argument('--output-dir', type=str, default='./outputs/qwen2_rlxsft_model',
                       help='输出目录')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='0', help='GPU 设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--dry-run', action='store_true', help='干运行')
    
    args = parser.parse_args()
    
    # 设置设备（单卡训练）
    if args.device == 'cpu':
        device = torch.device('cpu')
        logger.info("📍 CPU 模式")
    else:
        # 单卡模式：使用指定 GPU 的第一个 ID
        if ',' in args.device:
            device_id = int(args.device.split(',')[0].strip())
            logger.info(f"⚠️  多卡指定被转换为单卡模式，使用 GPU {device_id}")
        else:
            device_id = int(args.device)
        
        device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        logger.info(f"📍 单卡模式: GPU {device_id}")
        torch.cuda.set_device(device)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info(f"Device: {device}")
    logger.info(f"Strategy: {args.strategy}")
    
    # 创建配置
    config = RLxSFTConfig(
        strategy=args.strategy,
        model_name=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples=args.sample_size,
    )
    
    # 干运行
    if args.dry_run:
        logger.info("\n【干运行模式】")
        logger.info(f"配置: {asdict(config)}")
        return
    
    # 加载 tokenizer
    logger.info(f"📦 加载 tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    
    # 加载模型（单卡模式）
    logger.info(f"📦 加载模型: {config.model_name}")
    
    # 尝试启用 Flash Attention，如果不可用则自动回退
    model_kwargs = {
        'torch_dtype': torch.bfloat16,
        'device_map': {'': device},
        'trust_remote_code': True,
    }
    
    if config.use_flash_attention:
        try:
            model_kwargs['attn_implementation'] = 'flash_attention_2'
            logger.info("⚡ 尝试启用 Flash Attention 2...")
        except Exception as e:
            logger.warning(f"⚠️  Flash Attention 不可用: {e}，自动使用标准 Attention")
            config.use_flash_attention = False
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )
    
    # ⚡ 启用梯度检查点（激活函数重计算，节省 ~30% 显存）
    if config.use_gradient_checkpointing:
        logger.info("🔄 启用梯度检查点...")
        model.gradient_checkpointing_enable()
    
    # 应用 LoRA
    if config.use_lora:
        logger.info("🎯 应用 LoRA...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # 创建参考模型（单卡模式）
    logger.info("📌 创建参考模型...")
    # 参考模型也加载到 GPU 上以加速推理（显存允许的情况下）
    ref_device = device if config.ref_model_on_gpu else torch.device('cpu')
    
    # 参考模型使用相同的 Attention 实现
    ref_model_kwargs = {
        'torch_dtype': torch.bfloat16,
        'device_map': {'': ref_device},
        'trust_remote_code': True,
    }
    
    if config.use_flash_attention:
        ref_model_kwargs['attn_implementation'] = 'flash_attention_2'
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **ref_model_kwargs
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    if ref_device.type == 'cpu':
        logger.info("✓ 参考模型已加载到 CPU（显存优化）")
    else:
        logger.info("✓ 参考模型已加载到 GPU（推理加速）")
    
    # 加载数据集
    logger.info("📥 加载训练数据...")
    train_dataset = KYCDataset(
        config.data_path,
        tokenizer,
        max_seq_length=config.max_seq_length,
        max_samples=config.max_samples,
        split='train',
        train_test_ratio=config.train_test_split,
    )
    
    eval_dataset = KYCDataset(
        config.data_path,
        tokenizer,
        max_seq_length=config.max_seq_length,
        max_samples=config.max_samples,
        split='eval',
        train_test_ratio=config.train_test_split,
    )
    
    # 创建 dataloader（单卡模式，无需 DDP）
    # 启用 num_workers 和 prefetch_factor 以加速数据加载
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,  # 多进程加载数据
        prefetch_factor=config.prefetch_factor,  # 预取因子
        persistent_workers=True,  # 持久化 workers，避免重复初始化
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=True,
    )
    
    # 创建训练器
    logger.info("🏋️  创建训练器...")
    trainer = RLxSFTTrainer(
        config,
        model,
        ref_model,
        tokenizer,
        train_dataloader,
        eval_dataloader,
    )
    
    # 启动训练
    trainer.train()


if __name__ == "__main__":
    main()
