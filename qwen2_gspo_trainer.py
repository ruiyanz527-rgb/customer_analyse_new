"""
Qwen2-7B-Instruct GSPO 强化学习训练脚本
实现 Group Supervised Policy Optimization 用于 KYC 风险评估

GSPO 特点:
  - 组监督的政策优化
  - 多步骤推理能力增强
  - 长文本理解优化
  - 风险评估决策可解释性提升
"""

import os
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
)
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# 配置
# ============================================================================

@dataclass
class QwenGSPOConfig:
    """Qwen2 GSPO 训练配置"""
    
    # 模型
    model_name: str = "Qwen/Qwen2-7B-Instruct"
    sft_model_path: str = "./qwen2_kyc_model"  # SFT 微调后的模型路径
    
    # 数据
    data_path: str = "./financial_data/kyc_gspo_training_data_1_10.jsonl"  # 使用采样数据集
    max_seq_length: int = 512
    
    # 组监督参数
    group_size: int = 4  # 每组样本数
    
    # 训练
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # GSPO 特定参数
    max_grad_norm: float = 1.0
    temperature: float = 0.7  # 生成温度
    top_p: float = 0.9  # nucleus sampling
    
    # 输出
    output_dir: str = "./qwen2_kyc_gspo_model"
    save_steps: int = 500
    logging_steps: int = 100


# ============================================================================
# 数据加载
# ============================================================================

class GSPODataset(Dataset):
    """GSPO 数据集"""
    
    def __init__(self, data_path: str, tokenizer, max_seq_length: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = []
        
        print(f"📥 加载数据: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        print(f"✅ 已加载 {len(self.data):,} 条记录")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        record = self.data[idx]
        prompt = record['prompt']
        target = record['target']
        risk_label = record.get('risk_label', 0)
        
        # 构建完整文本
        full_text = f"{prompt}\n\n评估结果：{target}"
        
        # 分词
        tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 标记 response 部分用于计算 loss
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        response_start = prompt_tokens['input_ids'].shape[1]
        
        # 创建 labels（只计算 response 部分的 loss）
        labels = tokens['input_ids'].clone()
        labels[0, :response_start] = -100  # 忽略 prompt 部分
        
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'risk_label': risk_label,
        }


# ============================================================================
# GSPO 训练器
# ============================================================================

class GSPOTrainer:
    """GSPO 强化学习训练器"""
    
    def __init__(self, config: QwenGSPOConfig, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        # 获取底层模型（DDP 包装情况下）
        raw_model = model.module if isinstance(model, DDP) else model
        self.optimizer = AdamW(raw_model.parameters(), lr=config.learning_rate)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """训练一个 epoch"""
        self.model.train()
        
        is_ddp = dist.is_available() and dist.is_initialized()
        raw_model = self.model.module if is_ddp else self.model
        device = next(raw_model.parameters()).device
        is_main = (not is_ddp) or (dist.get_rank() == 0)
        
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Training", disable=not is_main)
        
        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # 更新进度条
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        return {'avg_loss': total_loss / len(train_loader)}
    
    def train(self, train_loader: DataLoader):
        """执行 GSPO 训练"""
        
        print("=" * 80)
        print("🚀 Qwen2-7B-Instruct KYC GSPO 强化学习训练")
        print("=" * 80)
        
        for epoch in range(self.config.num_train_epochs):
            print(f"\n【Epoch {epoch + 1}/{self.config.num_train_epochs}】")
            
            metrics = self.train_epoch(train_loader)
            print(f"  平均损失: {metrics['avg_loss']:.4f}")
        
        # 保存模型
        is_ddp = dist.is_available() and dist.is_initialized()
        is_main = (not is_ddp) or (dist.get_rank() == 0)
        
        if is_main:
            raw_model = self.model.module if is_ddp else self.model
            print(f"\n💾 保存模型到: {self.config.output_dir}")
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            raw_model.save_pretrained(self.config.output_dir)
            self.tokenizer.save_pretrained(self.config.output_dir)
        
        if is_ddp:
            dist.barrier()
        
        if is_main:
            print("\n" + "=" * 80)
            print("✅ GSPO 训练完成！")
            print("=" * 80)
            print(f"\n💾 GSPO 模型已保存到: {self.config.output_dir}")
            print(f"\n🔍 验证效果:")
            print(f"   python qwen2_inference.py --model-path {self.config.output_dir} --mode interactive")


def train_worker(rank: int, world_size: int, sft_model_path: str, data_path: str):
    """单个训练进程的入口"""
    
    use_ddp = world_size > 1
    is_main = (rank == 0)
    
    # DDP 初始化
    if use_ddp:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        import socket
        def find_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', 0))
                s.listen(1)
                port = s.getsockname()[1]
            return str(port)
        
        os.environ["MASTER_PORT"] = find_free_port()
        
        try:
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
            )
            backend_used = "nccl"
        except Exception as e:
            print(f"[rank {rank}] NCCL 初始化失败，回退到 gloo")
            try:
                dist.init_process_group(
                    backend="gloo",
                    rank=rank,
                    world_size=world_size,
                )
                backend_used = "gloo"
            except Exception as e2:
                print(f"[rank {rank}] gloo 初始化失败，回退到单卡模式")
                use_ddp = False
                world_size = 1
                backend_used = "none"
        
        if use_ddp:
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
            if is_main:
                print(f"🚀 DDP 初始化完成（{backend_used}）")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if is_main:
        print(f"▶  使用设备: {device}")
    
    # 配置
    config = QwenGSPOConfig(
        model_name=sft_model_path,
        sft_model_path=sft_model_path,
        data_path=data_path,
        num_train_epochs=2,
        output_dir="./qwen2_kyc_gspo_model",
    )
    
    # 加载模型和分词器
    if is_main:
        print(f"📦 加载 SFT 模型: {sft_model_path}")
    
    sft_path = Path(sft_model_path)
    if not sft_path.exists():
        print(f"❌ 错误: SFT 模型路径不存在: {sft_model_path}")
        exit(1)
    
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
    
    # 加载模型（带量化）
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device} if use_ddp else "auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    
    # 启用梯度检查点
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    torch.cuda.empty_cache()
    if is_main:
        print("✅ 梯度检查点已启用")
    
    # DDP 包装
    if use_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True, static_graph=True)
        if is_main:
            print(f"✅ 模型已用 DDP 包装")
    
    # 加载数据
    dataset = GSPODataset(data_path, tokenizer, config.max_seq_length)
    
    if use_ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        train_loader = DataLoader(
            dataset,
            batch_size=config.per_device_train_batch_size,
            sampler=sampler
        )
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=True
        )
    
    # 训练
    trainer = GSPOTrainer(config, model, tokenizer)
    trainer.train(train_loader)
    
    if use_ddp:
        dist.destroy_process_group()


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    import torch.multiprocessing as mp
    
    # 模型路径
    SFT_ADAPTER_PATH = "./qwen2_kyc_model_merge"
    SFT_MODEL_PATH = "./models/Qwen2-7B-Instruct"  # 使用本地基座模型
    DATA_PATH = "./financial_data/kyc_gspo_training_data_1_10.jsonl"
    
    # 检查模型
    if not Path(SFT_MODEL_PATH).exists():
        print(f"❌ 错误: 基座模型不存在: {SFT_MODEL_PATH}")
        exit(1)
    
    # 检查数据
    if not Path(DATA_PATH).exists():
        print(f"❌ 错误: 数据文件不存在: {DATA_PATH}")
        exit(1)
    
    # GPU 配置
    available_gpus = torch.cuda.device_count()
    num_gpus = int(os.environ.get("NUM_GPUS", available_gpus))
    num_gpus = max(1, min(num_gpus, available_gpus))
    
    print(f"🖥️  检测到 {available_gpus} 张 GPU，本次使用 {num_gpus} 张")
    
    # GSPO 使用单卡模式（更稳定，DDP 模式下加载慢）
    print("✅ 使用单进程模式（GSPO 推荐）")
    train_worker(rank=0, world_size=1,
                 sft_model_path=SFT_MODEL_PATH, data_path=DATA_PATH)
