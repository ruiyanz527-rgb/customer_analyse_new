"""
课程学习（Curriculum Learning）训练框架 - 完整版
Stage 1: 简单样本训练（使用已有 SFT 模型）
Stage 2: 复杂样本训练（使用复杂数据集混合训练）
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

import sys
import json
import csv
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


# ============================================================================
# 工具函数
# ============================================================================

def clear_gpu_memory():
    """清空 GPU 显存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def verify_model_path(model_path: str, model_name: str = "模型") -> Path:
    """验证模型路径是否存在"""
    model_path_obj = Path(model_path)
    
    if not model_path_obj.exists() and not model_path_obj.is_absolute():
        model_path_obj = Path.cwd() / model_path
    
    if not model_path_obj.exists():
        print(f"\n❌ 错误: {model_name}路径不存在: {model_path}")
        print(f"   完整路径: {model_path_obj.resolve()}")
        raise FileNotFoundError(f"找不到{model_name}: {model_path}")
    
    return model_path_obj


# ============================================================================
# 配置
# ============================================================================

@dataclass
class Config:
    """训练配置"""
    # 基础配置
    base_model: str = "models/Qwen2-7B-Instruct"
    data_path: str = "financial_data/kyc_gspo_training_data_1_10.jsonl"
    complex_data_path: str = "financial_data/kyc_rl_training_dataset_with_complex_text.csv"
    output_dir: str = "curriculum_model"
    sft_model: str = "qwen2_kyc_model_merged_full"
    
    # 训练参数（已优化）
    max_seq_len: int = 128
    stage1_epochs: int = 1
    stage1_lr: float = 1e-4
    stage1_batch: int = 1
    stage2_epochs: int = 1
    stage2_lr: float = 1e-4
    stage2_batch: int = 1
    
    # 输出路径
    easy_samples: str = "curriculum_model/easy_samples.jsonl"
    hard_samples: str = "curriculum_model/hard_samples.jsonl"
    stage1_model: str = "curriculum_model/stage1"
    stage2_model: str = "curriculum_model/final"
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# 数据集类
# ============================================================================

class KYCDataset(Dataset):
    """KYC 样本数据集（JSONL 格式）"""
    
    def __init__(self, data: List[Dict], tokenizer, max_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get('prompt', '')
        target = item.get('target', '')
        text = f"{prompt}\n{target}"
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
        }


class ComplexKYCDataset(Dataset):
    """复杂 KYC 数据集（CSV 格式）"""
    
    def __init__(self, csv_path, tokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        
        print(f"📥 从 CSV 加载数据：{csv_path}")
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # 获取列名
                if reader.fieldnames:
                    print(f"   列名: {reader.fieldnames}")
                
                for row in reader:
                    if row is None:
                        continue
                    
                    # 自动识别列名
                    prompt = (row.get('prompt') or row.get('instruction') or 
                             row.get('text') or row.get('input') or '')
                    target = (row.get('target') or row.get('output') or 
                             row.get('response') or row.get('answer') or '')
                    
                    # 如果没找到标准列名，尝试用前两列
                    if not prompt or not target:
                        vals = [v for v in row.values() if v]
                        if len(vals) >= 2:
                            prompt = vals[0] or ''
                            target = vals[1] or ''
                    
                    if prompt and target:
                        self.data.append({
                            'prompt': str(prompt).strip(),
                            'target': str(target).strip()
                        })
        
        except Exception as e:
            print(f"❌ CSV 加载异常: {e}")
            raise
        
        print(f"✅ 加载了 {len(self.data)} 条复杂样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"{item['prompt']}\n{item['target']}"
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
        }


class MixedDataset(Dataset):
    """混合简单和复杂样本的数据集"""
    
    def __init__(self, easy_samples, complex_dataset, tokenizer, max_len=128, ratio=1.0):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # 按比例混合
        num_complex = len(complex_dataset.data)
        num_easy = int(num_complex * ratio)
        self.easy_samples = easy_samples[:num_easy]
        self.complex_data = complex_dataset.data
        
        self.mixed_data = self.easy_samples + self.complex_data
        
        print(f"\n📊 混合数据构成:")
        print(f"  - 简单样本: {len(self.easy_samples)}")
        print(f"  - 复杂样本: {len(self.complex_data)}")
        print(f"  - 总计: {len(self.mixed_data)}")
    
    def __len__(self):
        return len(self.mixed_data)
    
    def __getitem__(self, idx):
        item = self.mixed_data[idx]
        prompt = item.get('prompt', '')
        target = item.get('target', '')
        text = f"{prompt}\n{target}"
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
        }


# ============================================================================
# 难度评估与样本划分
# ============================================================================

def estimate_difficulty(sample: Dict) -> float:
    """评估单个样本的难度（基于文本特征）"""
    try:
        prompt = sample.get('prompt', '')
        target = sample.get('target', '')
        
        # 方法 1：基于文本长度
        total_len = len(prompt) + len(target)
        length_difficulty = min(total_len / 500.0, 1.0)
        
        # 方法 2：基于复杂度指标
        complexity_score = 0.0
        if len(target) > 100:
            complexity_score += 0.2
        if target.count('，') + target.count('。') + target.count('、') > 5:
            complexity_score += 0.2
        if any(word in target for word in ['风险', '违规', '监管', '复杂', '严重']):
            complexity_score += 0.3
        
        complexity_difficulty = min(complexity_score, 1.0)
        
        # 综合难度
        difficulty = length_difficulty * 0.4 + complexity_difficulty * 0.6
        difficulty = min(max(difficulty, 0.1), 0.9)
        
    except Exception as e:
        print(f"⚠️ 难度评估异常: {e}")
        difficulty = 0.5
    
    return difficulty


def split_samples(config: Config):
    """难度评估与样本划分"""
    print("\n" + "="*80)
    print("🎯 难度评估与样本划分")
    print("="*80)
    
    # 加载 tokenizer
    print(f"\n📦 加载 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.sft_model, trust_remote_code=True)
    except Exception as e:
        print(f"\n❌ 加载 tokenizer 失败: {e}")
        raise
    
    # 加载数据
    print(f"📥 加载数据...")
    data = []
    with open(config.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    
    # 评估难度
    print(f"📊 评估 {len(data)} 个样本的难度...")
    difficulties = []
    for idx, sample in enumerate(tqdm(data, desc="难度评估")):
        diff = estimate_difficulty(sample)
        difficulties.append(diff)
    
    # 划分样本
    threshold = np.median(difficulties)
    easy_samples = [s for s, d in zip(data, difficulties) if d <= threshold]
    hard_samples = [s for s, d in zip(data, difficulties) if d > threshold]
    
    print(f"\n✅ 样本划分完成:")
    print(f"  难度阈值: {threshold:.4f}")
    print(f"  简单样本: {len(easy_samples)} ({len(easy_samples)/len(data)*100:.1f}%)")
    print(f"  困难样本: {len(hard_samples)} ({len(hard_samples)/len(data)*100:.1f}%)")
    
    # 保存
    Path(config.easy_samples).parent.mkdir(parents=True, exist_ok=True)
    
    with open(config.easy_samples, 'w', encoding='utf-8') as f:
        for s in easy_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    
    with open(config.hard_samples, 'w', encoding='utf-8') as f:
        for s in hard_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    
    print(f"💾 已保存到: {config.easy_samples} / {config.hard_samples}")
    
    return easy_samples, hard_samples, tokenizer


# ============================================================================
# Stage 1: 简单样本训练
# ============================================================================

def train_stage1(config: Config, easy_samples: List[Dict], tokenizer):
    """Stage 1: 简单样本基础训练"""
    
    print("\n" + "="*80)
    print("🚀 Stage 1: 简单样本基础训练")
    print("="*80)
    
    clear_gpu_memory()
    
    # 加载模型
    print(f"\n📦 加载 SFT 模型: {config.sft_model}")
    verify_model_path(config.sft_model, "SFT 模型")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.sft_model,
            dtype=torch.float16,
            device_map="cuda:0",  # 直接放在第一个GPU上
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"\n❌ 加载模型失败: {e}")
        raise
    
    # 准备数据
    print(f"📊 准备数据...")
    dataset = KYCDataset(easy_samples, tokenizer, config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=config.stage1_batch, shuffle=True)
    
    # 训练
    model.train()
    optimizer = AdamW(model.parameters(), lr=config.stage1_lr)
    
    print(f"\n🔥 开始训练...")
    print(f"  样本数: {len(easy_samples)}")
    print(f"  学习率: {config.stage1_lr}")
    print(f"  Epochs: {config.stage1_epochs}")
    
    total_loss = 0
    for epoch in range(config.stage1_epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.stage1_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to('cuda:0')
            attention_mask = batch['attention_mask'].to('cuda:0')
            
            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                
                if loss is None or torch.isnan(loss):
                    continue
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{epoch_loss/(batch_idx+1):.4f}'})
                
                if (batch_idx + 1) % 10 == 0:
                    del outputs, loss
                    clear_gpu_memory()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️  显存不足，清理后重试...")
                    clear_gpu_memory()
                else:
                    raise
        
        total_loss = epoch_loss / len(dataloader)
    
    print(f"\n✅ Stage 1 训练完成! 最终损失: {total_loss:.4f}")
    
    # 保存
    print(f"💾 保存模型到: {config.stage1_model}")
    Path(config.stage1_model).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.stage1_model)
    tokenizer.save_pretrained(config.stage1_model)
    print(f"✅ 模型保存成功")


# ============================================================================
# Stage 2: 复杂样本混合训练
# ============================================================================

def train_stage2(config: Config, easy_samples: List[Dict], tokenizer):
    """Stage 2: 复杂样本混合训练"""
    
    print("\n" + "="*80)
    print("🚀 Stage 2: 复杂样本混合训练")
    print("="*80)
    
    clear_gpu_memory()
    
    # 验证文件
    print(f"\n📋 验证文件...")
    verify_model_path(config.stage1_model, "Stage 1 模型")
    verify_model_path(config.complex_data_path, "复杂数据集")
    print(f"✅ 文件验证通过")
    
    # 加载 Stage 1 模型
    print(f"\n📦 加载 Stage 1 模型...")
    model = AutoModelForCausalLM.from_pretrained(
        config.stage1_model,
        dtype=torch.float16,
        device_map="cuda:0",  # 直接放在第一个GPU上
        trust_remote_code=True,
    )
    
    # 加载复杂数据集
    print(f"\n📥 加载复杂数据集...")
    try:
        complex_dataset = ComplexKYCDataset(config.complex_data_path, tokenizer, config.max_seq_len)
        if len(complex_dataset.data) == 0:
            print(f"⚠️  警告: 复杂数据集为空，检查文件格式...")
            print(f"   文件路径: {config.complex_data_path}")
            print(f"   请确保 CSV 文件包含 'prompt' 和 'target' 列")
            return
    except Exception as e:
        print(f"❌ 加载复杂数据集失败: {e}")
        print(f"   文件路径: {config.complex_data_path}")
        raise
    
    # 混合数据集
    print(f"\n🔄 构建混合数据集...")
    mixed_dataset = MixedDataset(easy_samples, complex_dataset, tokenizer, config.max_seq_len, ratio=1.0)
    dataloader = DataLoader(mixed_dataset, batch_size=config.stage2_batch, shuffle=True)
    
    # 训练
    model.train()
    optimizer = AdamW(model.parameters(), lr=config.stage2_lr)
    
    print(f"\n🔥 开始训练...")
    print(f"  学习率: {config.stage2_lr}")
    print(f"  Epochs: {config.stage2_epochs}")
    
    total_loss = 0
    for epoch in range(config.stage2_epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.stage2_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to('cuda:0')
            attention_mask = batch['attention_mask'].to('cuda:0')
            
            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                
                if loss is None or torch.isnan(loss):
                    continue
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{epoch_loss/(batch_idx+1):.4f}'})
                
                if (batch_idx + 1) % 10 == 0:
                    del outputs, loss
                    clear_gpu_memory()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n⚠️  显存不足，清理后重试...")
                    clear_gpu_memory()
                else:
                    raise
        
        total_loss = epoch_loss / len(dataloader)
    
    print(f"\n✅ Stage 2 训练完成! 最终损失: {total_loss:.4f}")
    
    # 保存
    print(f"💾 保存模型到: {config.stage2_model}")
    Path(config.stage2_model).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.stage2_model)
    tokenizer.save_pretrained(config.stage2_model)
    print(f"✅ 模型保存成功")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="课程学习训练 - 完整版")
    parser.add_argument('--skip-split', action='store_true', help='跳过难度划分')
    parser.add_argument('--skip-stage1', action='store_true', help='跳过 Stage 1')
    parser.add_argument('--skip-stage2', action='store_true', help='跳过 Stage 2')
    parser.add_argument('--output', default='curriculum_model', help='输出目录')
    parser.add_argument('--sft-model', default='qwen2_kyc_model_merged_full', help='SFT 模型路径')
    parser.add_argument('--complex-data', default='financial_data/kyc_rl_training_dataset_with_complex_text.csv', help='复杂数据集路径')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎓 课程学习 (Curriculum Learning) 训练 - 完整版")
    print("="*80)
    
    config = Config()
    config.output_dir = args.output.lstrip('./')
    config.sft_model = args.sft_model.lstrip('./')
    config.complex_data_path = args.complex_data.lstrip('./')
    config.easy_samples = f"{config.output_dir}/easy_samples.jsonl"
    config.hard_samples = f"{config.output_dir}/hard_samples.jsonl"
    config.stage1_model = f"{config.output_dir}/stage1"
    config.stage2_model = f"{config.output_dir}/final"
    
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📋 配置信息:")
    print(f"  SFT 模型: {config.sft_model}")
    print(f"  复杂数据: {config.complex_data_path}")
    print(f"  输出目录: {config.output_dir}")
    
    # Step 1: 难度划分
    if not args.skip_split:
        easy_samples, hard_samples, tokenizer = split_samples(config)
    else:
        print("\n⏭️  跳过难度划分，加载已有样本...")
        easy_samples = []
        if Path(config.easy_samples).exists():
            with open(config.easy_samples) as f:
                for line in f:
                    try:
                        easy_samples.append(json.loads(line))
                    except:
                        pass
        print(f"✅ 加载 {len(easy_samples)} 个简单样本")
        
        # 加载 tokenizer
        verify_model_path(config.sft_model, "SFT 模型")
        tokenizer = AutoTokenizer.from_pretrained(config.sft_model, trust_remote_code=True)
    
    # Stage 1
    if not args.skip_stage1 and easy_samples:
        train_stage1(config, easy_samples, tokenizer)
    else:
        print("\n⏭️  跳过 Stage 1")
    
    # Stage 2
    if not args.skip_stage2 and easy_samples:
        train_stage2(config, easy_samples, tokenizer)
    else:
        print("\n⏭️  跳过 Stage 2")
    
    print("\n" + "="*80)
    print("✅ 训练完成!")
    print("="*80)
    print(f"\n📁 输出目录: {config.output_dir}")
    print(f"  - Stage 1 模型: {config.stage1_model}")
    print(f"  - Stage 2 模型: {config.stage2_model} ⭐")
    print()


if __name__ == "__main__":
    main()
