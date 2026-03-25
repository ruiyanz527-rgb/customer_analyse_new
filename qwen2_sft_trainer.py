"""
Qwen2-7B-Instruct SFT微调训练脚本
用于KYC强化学习数据集的有监督微调

特性:
  - 支持长文本处理 (32K context)
  - 支持多GPU分布式训练
  - 集成Flash Attention优化
  - 完整的评估和保存机制
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
# import wandb  # 禁用wandb以避免protobuf冲突
from tqdm import tqdm

# ============================================================================
# 模型路径自动检测
# ============================================================================

def find_model_snapshot_path():
    """自动查找本地缓存的模型路径"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    qwen_dir = cache_dir / "models--Qwen--Qwen2-7B-Instruct"
    
    if not qwen_dir.exists():
        return None
    
    # 查找snapshots目录中的真实模型路径
    snapshots_dir = qwen_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = list(snapshots_dir.iterdir())
        if snapshots:
            # 返回第一个快照目录
            model_path = snapshots[0]
            # 验证关键文件是否存在
            if (model_path / "config.json").exists():
                return str(model_path)
    
    # 如果snapshots中没有，则直接返回qwen_dir（让transformers自己处理）
    return str(qwen_dir)

# ============================================================================
# 配置
# ============================================================================

@dataclass
class QwenSFTConfig:
    """Qwen2 SFT训练配置"""
    
    # 模型配置
    # 选项1：在线模式 (需要网络，会自动下载)
    # model_name: str = "Qwen/Qwen2-7B-Instruct"
    
    # 选项2：离线模式 (使用本地已下载的模型)
    model_name: str = "/root/.cache/huggingface/hub/models--Qwen--Qwen2-7B-Instruct"
    
    model_type: str = "qwen2"
    
    # 数据配置
    data_path: str = "/root/autodl-tmp/customer-analyse/financial_data/kyc_gspo_training_data.jsonl"
    max_seq_length: int = 2048
    preprocessing_num_workers: int = 4
    
    # 训练配置
    output_dir: str = "./qwen2_kyc_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # 优化配置
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    
    # 量化配置 (可选，节省显存)
    use_4bit: bool = False
    use_8bit: bool = False
    
    # 其他配置
    seed: int = 42
    device_map: str = "auto"
    use_flash_attention: bool = False  # 禁用Flash Attention 2（减少依赖）
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 100
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])  # 使用TensorBoard记录日志


# ============================================================================
# 数据加载和处理
# ============================================================================

class KYCDataProcessor:
    """KYC数据处理器"""
    
    def __init__(self, config: QwenSFTConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
    
    def load_data(self) -> DatasetDict:
        """加载JSONL格式的KYC数据"""
        print(f"📥 加载数据: {self.config.data_path}")
        
        # 加载JSONL文件
        data = []
        with open(self.config.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        print(f"✅ 已加载 {len(data):,} 条记录")
        
        # 转换为Dataset
        dataset = Dataset.from_dict({
            'user_id': [d['user_id'] for d in data],
            'prompt': [d['prompt'] for d in data],
            'target': [d['target'] for d in data],
            'text_length': [d['text_length'] for d in data],
            'risk_label': [d['risk_label'] for d in data],
        })
        
        # 分割数据集 (8:1:1)
        print("\n📊 分割数据集...")
        train_val = dataset.train_test_split(test_size=0.2, seed=self.config.seed)
        train_dataset = train_val['train']
        
        val_test = train_val['test'].train_test_split(test_size=0.5, seed=self.config.seed)
        val_dataset = val_test['train']
        test_dataset = val_test['test']
        
        print(f"  训练集: {len(train_dataset):,}")
        print(f"  验证集: {len(val_dataset):,}")
        print(f"  测试集: {len(test_dataset):,}")
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset,
        })
    
    def format_prompt(self, record: Dict) -> str:
        """格式化单条记录为模型输入"""
        prompt = record['prompt']
        target = record['target']
        
        # 构建完整序列
        formatted = f"{prompt}\n\n评估结果：{target}"
        return formatted
    
    def preprocess_function(self, examples):
        """预处理函数 - 手动处理labels，避免DataCollator问题"""
        
        # 格式化完整文本
        texts = []
        for prompt, target in zip(examples['prompt'], examples['target']):
            text = f"{prompt}\n\n评估结果：{target}"
            texts.append(text)
        
        # 分词
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding='max_length',
            return_tensors=None,
        )
        
        # 手动设置 labels：padding 位置设为 -100，其余位置与 input_ids 相同
        pad_id = self.tokenizer.pad_token_id
        labels = [
            [(token_id if token_id != pad_id else -100) for token_id in seq]
            for seq in tokenized['input_ids']
        ]
        tokenized['labels'] = labels
        
        return tokenized
    
    def process_data(self, dataset: DatasetDict) -> DatasetDict:
        """处理数据集"""
        print("\n🔨 预处理数据...")
        
        processed = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.config.preprocessing_num_workers,
            remove_columns=['user_id', 'prompt', 'target', 'text_length', 'risk_label'],
        )
        
        return processed


# ============================================================================
# 模型加载和准备
# ============================================================================

class QwenModelLoader:
    """Qwen2模型加载器"""
    
    def __init__(self, config: QwenSFTConfig):
        self.config = config
    
    def load_tokenizer(self):
        """加载分词器"""
        print(f"📦 加载分词器: {self.config.model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side='right',
            use_fast=True,
        )
        
        # 设置pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def load_model(self):
        """加载模型"""
        print(f"📦 加载模型: {self.config.model_name}")
        
        # 量化配置
        bnb_config = None
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.use_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,  # 使用bf16，比fp16更稳定，不容易NaN
            device_map=self.config.device_map,
            quantization_config=bnb_config if self.config.use_4bit or self.config.use_8bit else None,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if self.config.use_flash_attention else "eager",
        )
        
        # 应用LoRA
        if self.config.use_lora:
            print("🎯 应用LoRA优化...")
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        return model


# ============================================================================
# 训练
# ============================================================================

def train_qwen2_sft(config: QwenSFTConfig):
    """执行Qwen2 SFT训练"""
    
    print("=" * 80)
    print("🚀 Qwen2-7B-Instruct KYC强化学习SFT训练")
    print("=" * 80)
    
    # wandb已禁用，仅使用本地日志
    import os
    os.environ['WANDB_DISABLED'] = 'true'
    print("ℹ️  Weights & Biases已禁用，使用本地TensorBoard记录日志")
    
    # 加载模型和分词器
    loader = QwenModelLoader(config)
    tokenizer = loader.load_tokenizer()
    model = loader.load_model()
    
    print(f"\n✅ 模型加载完成")
    print(f"   参数量: {model.num_parameters() / 1e9:.2f}B")
    
    # 加载和处理数据
    processor = KYCDataProcessor(config, tokenizer)
    raw_dataset = processor.load_data()
    processed_dataset = processor.process_data(raw_dataset)
    
    print(f"\n✅ 数据处理完成")
    print(f"   输入长度: {config.max_seq_length}")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        
        # 保存和评估
        save_strategy="steps",
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        
        # 日志
        logging_steps=config.logging_steps,
        logging_dir='./logs',
        report_to=config.report_to,
        logging_nan_inf_filter=False,  # 不过滤NaN/Inf，显示真实损失值
        
        # 数值精度 - 使用bf16避免fp16的溢出问题
        bf16=True,
        fp16=False,
        
        # 其他
        seed=config.seed,
        dataloader_pin_memory=False,  # 8bit量化时关闭pin_memory
        remove_unused_columns=False,
        optim="adamw_torch",
    )
    
    # 使用默认 collator（labels 已在 preprocess 中处理好）
    data_collator = default_data_collator
    
    # 创建trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset['train'],
        eval_dataset=processed_dataset['validation'],
        data_collator=data_collator,
    )
    
    # 训练
    print("\n" + "=" * 80)
    print("🔥 开始训练...")
    print("=" * 80)
    
    train_result = trainer.train()
    
    print("\n" + "=" * 80)
    print("✅ 训练完成!")
    print("=" * 80)
    print(f"最终训练损失: {train_result.training_loss:.4f}")
    
    # 保存模型
    print(f"\n💾 保存模型到: {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # 测试集评估
    print(f"\n📊 在测试集上评估...")
    test_results = trainer.evaluate(processed_dataset['test'])
    print(f"测试集损失: {test_results['eval_loss']:.4f}")
    
    return model, tokenizer, trainer


# ============================================================================
# 推理
# ============================================================================

def inference_kyc_assessment(model, tokenizer, kyc_text: str, device='cuda'):
    """执行KYC风险评估推理"""
    
    model.eval()
    
    prompt = f"请基于以下KYC材料，完成多步骤的风险评估分析：\n\n{kyc_text}\n\n评估结果："
    
    # 分词
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1024,
            top_p=0.9,
            temperature=0.7,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取评估部分
    assessment_part = response.split("评估结果：")[-1] if "评估结果：" in response else response
    
    return assessment_part.strip()


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    # 自动检测本地模型路径
    local_model_path = find_model_snapshot_path()
    if local_model_path:
        print(f"✅ 检测到本地模型: {local_model_path}")
        model_name = local_model_path
    else:
        print("⚠️  未检测到本地模型，将尝试在线模式下载")
        model_name = "Qwen/Qwen2-7B-Instruct"
    
    # 创建配置（bf16无量化版本，解决NaN问题）
    config = QwenSFTConfig(
        model_name=model_name,
        num_train_epochs=3,
        max_seq_length=512,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,          # LoRA推荐更大的学习率
        use_lora=True,
        use_4bit=False,              # 禁用量化，避免NaN
        use_8bit=False,              # 禁用8bit量化，避免NaN
        lora_r=8,
        lora_alpha=16,
        use_flash_attention=False,
    )
    
    # 开始训练
    model, tokenizer, trainer = train_qwen2_sft(config)
    
    # ✅ 训练完成
    print("\n" + "=" * 80)
    print("✅ SFT训练完成！")
    print("=" * 80)
    print(f"\n💾 微调模型已保存到: {config.output_dir}")
    print(f"📊 TensorBoard日志已保存到: ./logs/")
    print(f"\n🔍 查看效果的方式:")
    print(f"  1. 查看TensorBoard: tensorboard --logdir=./logs/")
    print(f"  2. 运行推理: python qwen2_inference.py --model-path {config.output_dir} --mode interactive")
    print(f"  3. 运行评估: python qwen2_evaluate.py --model-path {config.output_dir}")
    print("\n" + "=" * 80)
    
    # 注释掉推理示例（避免CUDA设备错误）
    # sample_kyc = """【个人信息】..."""
    # assessment = inference_kyc_assessment(model, tokenizer, sample_kyc)