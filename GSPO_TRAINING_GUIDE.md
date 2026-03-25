# GSPO强化学习训练指南

## 概述

本指南展示如何利用生成的复杂长文本特征数据进行GLM-4-9B-Chat模型的GSPO（Group Supervised Policy Optimization）强化学习训练。

---

## 数据准备

### 1. 数据加载

```python
import pandas as pd
import json
import numpy as np
from pathlib import Path

# 加载CSV数据（包含长文本特征）
csv_path = '/Applications/financial LLM/financial_data/kyc_rl_training_dataset_with_complex_text.csv'
df = pd.read_csv(csv_path)

print(f"数据集大小: {len(df)} 条记录")
print(f"字段列表: {df.columns.tolist()}")

# 加载JSONL数据（GSPO格式）
jsonl_path = '/Applications/financial LLM/financial_data/kyc_gspo_training_data.jsonl'
train_data = []
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        train_data.append(json.loads(line))

print(f"GSPO训练集大小: {len(train_data)} 条记录")
```

### 2. 数据集分割

```python
from sklearn.model_selection import train_test_split

# 按风险标签分层抽样
low_risk_data = [d for d in train_data if d['risk_label'] == 0]
high_risk_data = [d for d in train_data if d['risk_label'] == 1]

print(f"低风险样本: {len(low_risk_data)}")
print(f"高风险样本: {len(high_risk_data)}")

# 按比例分割（8:1:1）
train_data_split = []
val_data_split = []
test_data_split = []

for data_group in [low_risk_data, high_risk_data]:
    train, temp = train_test_split(data_group, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    train_data_split.extend(train)
    val_data_split.extend(val)
    test_data_split.extend(test)

print(f"训练集: {len(train_data_split)} | 验证集: {len(val_data_split)} | 测试集: {len(test_data_split)}")
```

---

## 模型训练

### 1. 基础模型加载

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 使用GLM-4-9B-Chat作为基础模型
model_name = 'THUDM/glm-4-9b-chat'

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)

# 启用梯度计算
model.train()

print(f"模型加载完成")
print(f"模型参数量: {model.num_parameters() / 1e9:.2f}B")
```

### 2. 数据预处理

```python
class KYCGSPODataset(torch.utils.data.Dataset):
    """GSPO训练数据集"""
    
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        record = self.data[idx]
        
        # 构建输入序列
        prompt = record['prompt']
        target = record['target']
        
        # 拼接成完整序列
        full_text = f"请完成以下KYC风险评估：\n{prompt}\n评估结果：{target}"
        
        # 分词
        tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 标记target部分用于计算loss
        target_start = len(
            self.tokenizer(
                f"请完成以下KYC风险评估：\n{prompt}\n评估结果：",
                return_tensors='pt'
            )['input_ids'][0]
        )
        
        labels = tokens['input_ids'].clone()
        labels[0, :target_start] = -100  # 忽略prompt部分的loss
        
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

# 创建数据集
train_dataset = KYCGSPODataset(train_data_split, tokenizer)
val_dataset = KYCGSPODataset(val_data_split, tokenizer)
test_dataset = KYCGSPODataset(test_data_split, tokenizer)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4
)

print(f"训练集加载完成: {len(train_loader)} batches")
```

### 3. GSPO优化器配置

```python
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# 配置优化器
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# 配置学习率调度器
num_epochs = 3
num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = len(train_loader)  # 1个epoch的warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

print(f"训练步数: {num_training_steps}")
print(f"Warmup步数: {num_warmup_steps}")
```

### 4. GSPO训练循环

```python
from tqdm import tqdm
import wandb

# 初始化wandb记录
wandb.init(project="kyc-gspo-training", name="glm4-9b-chat-v1")

# 训练配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

best_val_loss = float('inf')
patience = 3
patience_counter = 0

# 训练循环
for epoch in range(num_epochs):
    print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
    
    # ===== 训练阶段 =====
    model.train()
    train_loss = 0
    train_pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(train_pbar):
        # 将数据移到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        avg_train_loss = train_loss / (batch_idx + 1)
        
        # 更新进度条
        train_pbar.set_postfix({'loss': f'{avg_train_loss:.4f}'})
        
        # 日志记录
        if batch_idx % 10 == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/avg_loss': avg_train_loss,
                'train/learning_rate': scheduler.get_last_lr()[0],
                'epoch': epoch
            })
    
    # ===== 验证阶段 =====
    model.eval()
    val_loss = 0
    val_pbar = tqdm(val_loader, desc="Validation")
    
    with torch.no_grad():
        for batch in val_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / (len(val_loader))
            val_pbar.set_postfix({'loss': f'{avg_val_loss:.4f}'})
    
    avg_val_loss = val_loss / len(val_loader)
    
    # 日志记录
    wandb.log({
        'val/loss': avg_val_loss,
        'epoch': epoch
    })
    
    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # 早停机制
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        
        # 保存最佳模型
        model.save_pretrained(f'./checkpoints/glm4-gspo-epoch{epoch+1}')
        tokenizer.save_pretrained(f'./checkpoints/glm4-gspo-epoch{epoch+1}')
        print(f"✓ 保存最佳模型 (Val Loss: {avg_val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"早停触发，停止训练")
            break

wandb.finish()
print("✓ 训练完成")
```

---

## 评估与测试

### 1. 测试集评估

```python
from nltk.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

def evaluate_on_test_set(model, test_loader, tokenizer, device):
    """在测试集上评估模型性能"""
    
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_losses = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 计算loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            all_losses.append(outputs.loss.item())
            
            # 生成预测
            with torch.cuda.amp.autocast():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=512,
                    top_p=0.9,
                    temperature=0.7,
                    do_sample=True
                )
            
            # 解码预测和目标
            pred_text = tokenizer.batch_decode(generated, skip_special_tokens=True)
            target_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            all_predictions.extend(pred_text)
            all_targets.extend(target_text)
    
    # 计算指标
    avg_loss = np.mean(all_losses)
    
    # BLEU分数
    bleu_scores = []
    for pred, target in zip(all_predictions, all_targets):
        pred_tokens = pred.split()
        target_tokens = target.split()
        bleu = sentence_bleu([target_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        bleu_scores.append(bleu)
    
    avg_bleu = np.mean(bleu_scores)
    
    # ROUGE分数
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    for pred, target in zip(all_predictions, all_targets):
        scores = scorer.score(target, pred)
        rouge_scores.append(scores['rougeL'].fmeasure)
    
    avg_rouge = np.mean(rouge_scores)
    
    return {
        'test_loss': avg_loss,
        'bleu_score': avg_bleu,
        'rouge_score': avg_rouge,
        'predictions': all_predictions,
        'targets': all_targets
    }

# 运行评估
test_results = evaluate_on_test_set(model, test_loader, tokenizer, device)

print(f"\n===== 测试集结果 =====")
print(f"Test Loss: {test_results['test_loss']:.4f}")
print(f"BLEU Score: {test_results['bleu_score']:.4f}")
print(f"ROUGE Score: {test_results['rouge_score']:.4f}")
```

### 2. 风险评估准确性评估

```python
def evaluate_risk_assessment_accuracy(model, test_data, tokenizer, device):
    """评估风险评估决策的准确性"""
    
    model.eval()
    
    correct_decisions = 0
    total_decisions = 0
    
    with torch.no_grad():
        for record in tqdm(test_data, desc="Evaluating Risk Assessment"):
            prompt = record['prompt']
            target = record['target']
            true_label = record['risk_label']
            
            # 生成预测
            input_text = f"请完成以下KYC风险评估：\n{prompt}\n评估结果："
            input_ids = tokenizer(input_text, return_tensors='pt').to(device)
            
            generated = model.generate(
                **input_ids,
                max_length=512,
                top_p=0.9,
                temperature=0.7,
                do_sample=False
            )
            
            pred_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            
            # 提取风险标签
            if '低风险' in pred_text or 'low risk' in pred_text.lower():
                pred_label = 0
            elif '高风险' in pred_text or 'high risk' in pred_text.lower():
                pred_label = 1
            else:
                continue  # 无法解析，跳过
            
            total_decisions += 1
            if pred_label == true_label:
                correct_decisions += 1
    
    accuracy = correct_decisions / total_decisions if total_decisions > 0 else 0
    
    return {
        'risk_assessment_accuracy': accuracy,
        'correct_decisions': correct_decisions,
        'total_decisions': total_decisions
    }

# 运行风险评估准确性评估
risk_results = evaluate_risk_assessment_accuracy(model, test_data_split, tokenizer, device)

print(f"\n===== 风险评估准确性 =====")
print(f"准确率: {risk_results['risk_assessment_accuracy']:.4f}")
print(f"正确决策: {risk_results['correct_decisions']}/{risk_results['total_decisions']}")
```

---

## 推理与部署

### 1. 交互式推理

```python
def inference_kyc_assessment(model, tokenizer, kyc_text, device):
    """执行KYC风险评估推理"""
    
    model.eval()
    
    prompt = f"请基于以下KYC材料，完成多步骤的风险评估分析：\n\n{kyc_text}\n\n评估结果："
    
    # 分词
    input_ids = tokenizer(prompt, return_tensors='pt').to(device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            max_length=1024,
            top_p=0.95,
            temperature=0.7,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取推理链路
    assessment_part = response.split("评估结果：")[-1] if "评估结果：" in response else response
    
    return assessment_part

# 示例：对新用户进行风险评估
sample_kyc = df['kyc_complex_text'].iloc[100]
assessment = inference_kyc_assessment(model, tokenizer, sample_kyc, device)

print("KYC材料:")
print(sample_kyc[:200] + "...")
print("\n评估结果:")
print(assessment)
```

### 2. 批量推理

```python
def batch_inference_kyc_assessment(model, tokenizer, kyc_texts, batch_size=4, device='cuda'):
    """批量执行KYC风险评估"""
    
    model.eval()
    results = []
    
    # 分批处理
    for i in range(0, len(kyc_texts), batch_size):
        batch_texts = kyc_texts[i:i+batch_size]
        
        # 构建批量输入
        prompts = [
            f"请基于以下KYC材料，完成多步骤的风险评估分析：\n\n{text}\n\n评估结果："
            for text in batch_texts
        ]
        
        # 分词
        inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors='pt'
        ).to(device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                top_p=0.95,
                temperature=0.7,
                do_sample=True
            )
        
        # 解码
        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(batch_results)
        
        print(f"已处理 {i+len(batch_texts)}/{len(kyc_texts)} 条记录")
    
    return results

# 批量推理
kyc_samples = df['kyc_complex_text'].head(100).tolist()
assessments = batch_inference_kyc_assessment(model, tokenizer, kyc_samples)

# 保存结果
results_df = pd.DataFrame({
    'kyc_text': kyc_samples,
    'assessment': assessments
})
results_df.to_csv('kyc_assessments.csv', index=False, encoding='utf-8')
print("✓ 推理结果已保存")
```

---

## 常见问题与优化

### 1. 内存优化

```python
# 启用梯度检查点
model.gradient_checkpointing_enable()

# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 在训练循环中使用
with autocast():
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. 分布式训练

```python
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group

# 初始化分布式训练
init_process_group(backend='nccl')

# 包装模型
model = DDP(model)

# 使用DistributedSampler
from torch.utils.data.distributed import DistributedSampler

train_sampler = DistributedSampler(train_dataset, shuffle=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    sampler=train_sampler
)
```

### 3. 长文本处理优化

```python
# 使用长文本优化的tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    model_max_length=2048,
    trust_remote_code=True
)

# 使用滑动窗口处理超长文本
def process_long_text(text, max_length=2048, stride=512):
    """使用滑动窗口处理超长文本"""
    tokens = tokenizer(text)['input_ids']
    
    segments = []
    for i in range(0, len(tokens) - max_length + 1, stride):
        segment = tokens[i:i+max_length]
        segments.append(tokenizer.decode(segment))
    
    return segments
```

---

## 总结

本指南演示了如何：
1. ✅ 加载和预处理GSPO格式的KYC数据
2. ✅ 使用GLM-4-9B-Chat进行GSPO强化学习训练
3. ✅ 评估模型在长文本推理中的性能
4. ✅ 进行KYC风险评估推理
5. ✅ 优化内存和计算效率

**预期收益**：
- 🎯 长文本理解能力提升
- 🎯 多步骤推理准确性提升
- 🎯 风险评估决策可解释性增强
- 🎯 金融应用落地能力强化

---

**参考资源**：
- [GLM-4文档](https://github.com/THUDM/GLM-4)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [GRPO论文](https://arxiv.org/abs/2402.03300)
