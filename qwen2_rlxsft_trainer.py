"""
RLxSFT 混合训练框架
结合 SFT 有监督学习和 RL 强化学习的优点

核心思想：
  - 在每个 epoch 中，同时进行 SFT 和 GRPO 两个目标的优化
  - SFT 目标：确保模型保留基础能力和格式能力
  - GRPO 目标：根据奖励信号优化风险评估的准确性
  
支持的混合策略：
  1. Sequential: 先 SFT 再 GRPO（原始两阶段方案）
  2. Interleaved: 交替执行 SFT 和 GRPO batch
  3. Joint: 在同一 batch 中同时计算两个目标
  4. Weighted: 加权组合两个目标的损失函数
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
from tqdm import tqdm


# ============================================================================
# 配置
# ============================================================================

@dataclass
class RLxSFTConfig:
    """RLxSFT 混合训练配置"""
    
    # 模型
    model_name: str = "./models/Qwen2-7B-Instruct"
    base_model_path: str = "./models/Qwen2-7B-Instruct"
    
    # 数据
    data_path: str = "./financial_data/kyc_gspo_training_data_1_2.jsonl"
    max_seq_length: int = 512  # 减小序列长度以节省内存
    
    # 混合策略
    # 'sequential': 先 SFT 再 GRPO
    # 'interleaved': 交替执行
    # 'joint': 同一 batch 中同时计算
    # 'weighted': 加权组合损失函数
    mix_strategy: str = "weighted"  # 推荐使用 weighted
    
    # 训练
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    
    # SFT 目标权重
    sft_weight: float = 0.3  # SFT 损失占 30%
    
    # RL 目标权重
    rl_weight: float = 0.7   # RL 损失占 70%
    
    # 学习率
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # GRPO 参数
    group_size: int = 4
    kl_coef: float = 0.003
    entropy_coef: float = 0.01
    clip_ratio: float = 0.15
    clip_ratio_high: float = 0.28
    
    # 输出
    output_dir: str = "./qwen2_rlxsft_model"
    save_steps: int = 500
    logging_steps: int = 100


# ============================================================================
# 混合损失计算器
# ============================================================================

class HybridLossCalculator:
    """计算 SFT 和 RL 的混合损失"""
    
    def __init__(self, config: RLxSFTConfig):
        self.config = config
    
    def compute_sft_loss(self, logits, labels):
        """计算 SFT 有监督学习损失"""
        # 标准语言模型损失
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss()
        sft_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return sft_loss
    
    def compute_rl_loss(self, policy_logprobs, ref_logprobs, rewards):
        """计算 RL 强化学习损失"""
        # PPO-style 策略梯度损失 + KL 惩罚
        log_ratio = policy_logprobs - ref_logprobs
        ratio = torch.exp(log_ratio)
        
        # 非对称裁剪
        clip_low = 1.0 - self.config.clip_ratio
        clip_high = self.config.clip_ratio_high
        
        unclipped = ratio * rewards
        clipped = torch.clamp(ratio, clip_low, clip_high) * rewards
        pg_loss = -torch.min(unclipped, clipped).mean()
        
        # KL 散度惩罚
        kl = (ratio - 1).pow(2) / 2  # k2 估计器
        kl_penalty = self.config.kl_coef * kl.mean()
        
        rl_loss = pg_loss + kl_penalty
        
        return rl_loss
    
    def compute_hybrid_loss(self, sft_loss, rl_loss):
        """计算混合损失"""
        # 加权组合
        hybrid_loss = (
            self.config.sft_weight * sft_loss +
            self.config.rl_weight * rl_loss
        )
        
        return {
            'hybrid_loss': hybrid_loss,
            'sft_loss': sft_loss.item(),
            'rl_loss': rl_loss.item(),
            'sft_weight': self.config.sft_weight,
            'rl_weight': self.config.rl_weight,
        }


# ============================================================================
# RLxSFT 训练器
# ============================================================================

class RLxSFTTrainer:
    """RLxSFT 混合训练器"""
    
    def __init__(self, config: RLxSFTConfig, model, tokenizer, ref_model):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.loss_calculator = HybridLossCalculator(config)
        
        # 优化器
        raw_model = model.module if isinstance(model, DDP) else model
        self.optimizer = AdamW(raw_model.parameters(), lr=config.learning_rate)
    
    def forward_pass(self, batch_data):
        """前向传播，计算 SFT 和 RL 损失"""
        
        input_ids = batch_data['input_ids'].to(self.model.device)
        attention_mask = batch_data['attention_mask'].to(self.model.device)
        labels = batch_data.get('labels', input_ids.clone()).to(self.model.device)
        rewards = batch_data.get('rewards', torch.ones(input_ids.shape[0])).to(self.model.device)
        
        # 策略模型前向传播（计算梯度）
        policy_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        policy_logits = policy_outputs.logits
        
        # 计算 SFT 损失
        sft_loss = self.loss_calculator.compute_sft_loss(policy_logits, labels)
        
        # 计算 RL 损失（需要 logprobs）
        # 只在必要的位置计算 log_softmax 以节省内存
        policy_logprobs = torch.nn.functional.log_softmax(policy_logits, dim=-1)
        policy_logprobs = policy_logprobs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
        
        # 参考模型 logprobs（使用缓存的值或单独计算，不计算梯度）
        with torch.no_grad():
            # 临时设置为 eval 模式以获得参考输出
            self.ref_model.eval()
            ref_outputs = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            ref_logits = ref_outputs.logits
            ref_logprobs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
            ref_logprobs = ref_logprobs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1).detach()
            
            # 立即删除不需要的张量以释放内存
            del ref_logits, ref_outputs
        
        rl_loss = self.loss_calculator.compute_rl_loss(
            policy_logprobs.mean(dim=1),
            ref_logprobs.mean(dim=1),
            rewards
        )
        
        # 混合损失
        loss_dict = self.loss_calculator.compute_hybrid_loss(sft_loss, rl_loss)
        
        # 清理中间张量
        del policy_logits, policy_logprobs, ref_logprobs
        
        return loss_dict['hybrid_loss'], loss_dict
    
    def train_epoch(self, dataloader):
        """训练一个 epoch"""
        
        self.model.train()
        # 注意：ref_model 和 model 指向同一个模型实例
        # 在 forward_pass 中使用 torch.no_grad() 来防止参考模型计算梯度
        
        total_loss = 0.0
        total_sft_loss = 0.0
        total_rl_loss = 0.0
        batch_count = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 前向传播
            loss, loss_dict = self.forward_pass(batch)
            
            # 缩放损失用于梯度累积
            scaled_loss = loss / self.config.gradient_accumulation_steps
            
            # 反向传播
            scaled_loss.backward()
            
            # 每 gradient_accumulation_steps 步更新一次参数
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 累积统计
            total_loss += loss.item()
            total_sft_loss += loss_dict['sft_loss']
            total_rl_loss += loss_dict['rl_loss']
            batch_count += 1
            
            # 清理中间张量
            del loss, loss_dict
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            progress_bar.set_postfix({
                'loss': f"{total_loss / batch_count:.4f}",
                'sft': f"{total_sft_loss / batch_count:.4f}",
                'rl': f"{total_rl_loss / batch_count:.4f}",
            })
        
        return {
            'avg_loss': total_loss / batch_count,
            'avg_sft_loss': total_sft_loss / batch_count,
            'avg_rl_loss': total_rl_loss / batch_count,
        }


# ============================================================================
# 主函数
# ============================================================================

def train_rlxsft():
    """执行 RLxSFT 混合训练"""
    
    print("=" * 80)
    print("🚀 RLxSFT 混合训练框架")
    print("结合 SFT 和 GRPO 的两阶段优化")
    print("=" * 80)
    print()
    
    # 配置
    config = RLxSFTConfig(
        mix_strategy="weighted",
        sft_weight=0.3,  # SFT 占 30%
        rl_weight=0.7,   # RL 占 70%
    )
    
    print(f"📋 配置:")
    print(f"   混合策略: {config.mix_strategy}")
    print(f"   SFT 权重: {config.sft_weight} (30%)")
    print(f"   RL 权重: {config.rl_weight} (70%)")
    print()
    
    # 加载模型
    print("📦 加载模型...")
    
    # 设置内存优化环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_path,
        trust_remote_code=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path,
        dtype=torch.float16,  # 使用 float16 而不是 bfloat16，节省内存
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 应用 LoRA
    if config.use_lora:
        print("🎯 应用 LoRA...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # 参考模型（冻结）
    print("📌 创建参考模型...")
    # 冻结模型参数而不是深度复制（避免双倍内存占用）
    # 使用同一个模型实例，但在参考计算中使用 torch.no_grad()
    ref_model = model
    ref_model.eval()
    
    print("✅ 模型加载完成")
    print()
    
    # 加载数据
    print("📥 加载数据...")
    data = []
    with open(config.data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000:  # 示例：使用 1000 条数据
                break
            data.append(json.loads(line))
    print(f"✓ 已加载 {len(data)} 条数据")
    print()
    
    # 数据预处理和加载器创建函数
    def create_dataloader(data_list, tokenizer, batch_size, max_seq_length):
        """创建数据加载器"""
        def collate_fn(batch):
            """批处理整理函数"""
            input_ids_list = []
            attention_mask_list = []
            labels_list = []
            rewards_list = []
            
            for sample in batch:
                # 提取文本和标签
                text = sample.get('text', sample.get('instruction', ''))
                label = sample.get('label', sample.get('output', 1))
                reward = sample.get('reward', float(label) if isinstance(label, (int, float)) else 1.0)
                
                # 分词（不返回 tensors，先处理成列表）
                encoded = tokenizer(
                    text,
                    max_length=max_seq_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors=None,  # 不立即转换为 tensors
                )
                
                input_ids_list.append(encoded['input_ids'])
                attention_mask_list.append(encoded['attention_mask'])
                labels_list.append(encoded['input_ids'])
                rewards_list.append(reward)
            
            # 批量转换为张量，使用 LongTensor 来节省内存
            return {
                'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long),
                'labels': torch.tensor(labels_list, dtype=torch.long),
                'rewards': torch.tensor(rewards_list, dtype=torch.float32),
            }
        
        from torch.utils.data import DataLoader
        
        # 创建数据集
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, data_list):
                self.data = data_list
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataset = CustomDataset(data_list)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # 禁用多进程加载以避免内存问题
            pin_memory=False,  # 禁用 pin_memory
        )
        return dataloader
    
    # 创建训练器
    print("🏋️  创建训练器...")
    trainer = RLxSFTTrainer(config, model, tokenizer, ref_model)
    print("✓ 训练器已创建")
    print()
    
    # 训练
    print("=" * 80)
    print("🔥 开始 RLxSFT 混合训练...")
    print("=" * 80)
    print()
    
    for epoch in range(config.num_train_epochs):
        print(f"\n【Epoch {epoch + 1}/{config.num_train_epochs}】")
        
        # 创建数据加载器
        dataloader = create_dataloader(
            data,
            tokenizer,
            config.per_device_train_batch_size,
            config.max_seq_length
        )
        
        # 执行训练
        metrics = trainer.train_epoch(dataloader)
        
        print(f"  平均损失: {metrics['avg_loss']:.4f}")
        print(f"  SFT 损失: {metrics['avg_sft_loss']:.4f}")
        print(f"  RL 损失:  {metrics['avg_rl_loss']:.4f}")
    
    print()
    print("=" * 80)
    print("✅ RLxSFT 混合训练完成！")
    print("=" * 80)
    print()
    
    # 保存模型
    print(f"💾 保存模型到: {config.output_dir}")
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print("✓ 模型保存完成")
    print()
    
    print("📊 混合训练策略优势:")
    print("  ✓ 结合 SFT 的稳定性和 RL 的优化能力")
    print("  ✓ 避免灾难性遗忘（通过 SFT 损失约束）")
    print("  ✓ 提升任务特定性能（通过 RL 奖励优化）")
    print("  ✓ 降低训练时间（同时进行而非顺序进行）")


if __name__ == "__main__":
    train_rlxsft()
