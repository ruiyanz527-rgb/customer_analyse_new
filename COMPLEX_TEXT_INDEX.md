# 复杂长文本特征 - 完整索引

## 📑 文档导航

### 🚀 快速开始
**→ 从这里开始** 👈

1. **COMPLEX_TEXT_QUICK_REFERENCE.md** (10KB)
   - 📌 快速参考卡片
   - ⏱️ 5分钟快速了解
   - 📊 数据统计、字段说明、使用示例
   - **推荐**: 第一次接触该功能的用户

2. **COMPLEX_TEXT_GENERATION_COMPLETE.md** (13KB)
   - 📌 完成报告，项目总结
   - ⏱️ 15分钟深入了解
   - 📈 成果展示、质量指标、应用场景
   - **推荐**: 项目经理、决策层

---

### 📚 详细文档

3. **COMPLEX_TEXT_FEATURES_SUMMARY.md** (15KB)
   - 📌 完整技术文档
   - ⏱️ 30分钟精读
   - 🔬 详细的文本结构、生成算法、应用指南
   - **推荐**: 技术实现、研究人员

4. **GSPO_TRAINING_GUIDE.md** (18KB)
   - 📌 实战训练指南
   - ⏱️ 45分钟学习
   - 💻 完整代码示例、训练循环、模型部署
   - **推荐**: 模型训练人员、NLP工程师

---

### 💾 数据文件

#### 训练数据

| 文件 | 大小 | 格式 | 用途 | 记录数 |
|------|------|------|------|--------|
| **kyc_rl_training_dataset_with_complex_text.csv** | 25MB | CSV | 完整训练集 | 10,000 |
| **kyc_gspo_training_data.jsonl** | 4MB | JSONL | GSPO格式 | 10,000 |

#### 数据字段说明

**kyc_rl_training_dataset_with_complex_text.csv**:
- 原始15列 + **kyc_complex_text** + **reasoning_chain_detailed**
- 可用pandas/SQL直接加载
- 包含所有原始特征 + 新增长文本特征

**kyc_gspo_training_data.jsonl**:
- 每行一条JSON记录
- 包含: user_id, prompt, target, text_length, risk_label
- 标准格式，易于流式处理

---

### 🔧 代码文件

| 文件 | 行数 | 功能 | 用途 |
|------|------|------|------|
| **generate_complex_kyc_texts.py** | 632 | 文本生成 | 复现数据生成过程 |

**主要类和方法**:
```python
ComplexKYCTextGenerator
  ├─ generate_work_experience()        # 生成工作背景
  ├─ generate_financial_status()       # 生成财务状况
  ├─ generate_credit_profile()         # 生成信用评分
  ├─ generate_risk_features()          # 生成风险评估
  ├─ generate_industry_analysis()      # 生成行业分析
  ├─ generate_comprehensive_kyc_text() # 生成综合长文本
  └─ generate_multi_step_reasoning()   # 生成推理链路
```

---

## 🎯 快速导航

### 我是...

#### 数据分析师
1. 阅读 **COMPLEX_TEXT_QUICK_REFERENCE.md** → 了解数据结构
2. 加载 **kyc_rl_training_dataset_with_complex_text.csv** → 开始分析
3. 参考 **COMPLEX_TEXT_FEATURES_SUMMARY.md** → 了解数据生成逻辑

#### 模型训练工程师
1. 快速浏览 **COMPLEX_TEXT_QUICK_REFERENCE.md** → 5分钟概览
2. 深入学习 **GSPO_TRAINING_GUIDE.md** → 复制训练代码
3. 加载 **kyc_gspo_training_data.jsonl** → 开始训练

#### 项目经理
1. 快速阅读 **COMPLEX_TEXT_GENERATION_COMPLETE.md** → 理解成果
2. 查看 **COMPLEX_TEXT_QUICK_REFERENCE.md** 中的统计数据
3. 关注预期效果部分 → 了解商业价值

#### 研究员
1. 深入阅读 **COMPLEX_TEXT_FEATURES_SUMMARY.md** → 理解方法
2. 运行 **generate_complex_kyc_texts.py** → 复现过程
3. 自定义参数 → 进行实验

---

## 📊 数据统计速览

### 基本信息
```
总记录数: 10,000条
低风险:   8,500条 (85%)
高风险:   1,500条 (15%)
文本大小: 25MB (CSV) + 4MB (JSONL)
```

### 文本长度
```
平均长度: 720字符
最小值:   600字符
最大值:   800字符
分布:     均匀分布
```

### 推理步骤
```
步骤数:   5-6步
格式:     [步骤名] 内容 | [步骤名] 内容 | ...
长度:     400-600字符
覆盖:     职业→收入→信用→交易→综合→决策
```

---

## 🔍 使用示例

### 方式1: 加载CSV数据
```python
import pandas as pd

# 加载数据
df = pd.read_csv('kyc_rl_training_dataset_with_complex_text.csv')

# 查看长文本特征
print(df['kyc_complex_text'].iloc[0])
print(df['reasoning_chain_detailed'].iloc[0])

# 统计
print(f"总数: {len(df)}")
print(f"风险分布:\n{df['is_risky_user'].value_counts()}")
```

### 方式2: 加载JSONL数据
```python
import json

# 加载GSPO格式数据
with open('kyc_gspo_training_data.jsonl') as f:
    for line in f:
        record = json.loads(line)
        print(f"User: {record['user_id']}")
        print(f"Prompt length: {record['text_length']}")
        print(f"Risk label: {record['risk_label']}")
```

### 方式3: 用于训练
```python
# 在GSPO训练中使用
dataset = []
with open('kyc_gspo_training_data.jsonl') as f:
    for line in f:
        record = json.loads(line)
        dataset.append({
            'input': record['prompt'],
            'target': record['target'],
            'label': record['risk_label']
        })

# 传入训练器
trainer.train(dataset)
```

---

## ⚡ 常见任务速查

### 任务1: 数据质量检查
**文档**: COMPLEX_TEXT_FEATURES_SUMMARY.md → "数据质量"部分
**时间**: 5分钟
**输出**: 质量报告

### 任务2: 模型训练
**文档**: GSPO_TRAINING_GUIDE.md → "模型训练"部分
**时间**: 2小时 (3个epoch)
**输出**: 训练好的模型

### 任务3: 性能评估
**文档**: GSPO_TRAINING_GUIDE.md → "评估与测试"部分
**时间**: 30分钟
**输出**: BLEU、ROUGE、准确率等指标

### 任务4: 推理部署
**文档**: GSPO_TRAINING_GUIDE.md → "推理与部署"部分
**时间**: 1小时
**输出**: 可在线推理的服务

---

## 📈 性能基准

### 数据生成性能
| 操作 | 耗时 | 速率 |
|------|------|------|
| 文本生成 | 240秒 | 42条/秒 |
| 推理链路 | 180秒 | 56条/秒 |
| 总耗时 | 462秒 | ~7.7分钟 |

### 模型性能期望
| 指标 | 基线 | 优化后 | 提升 |
|------|------|--------|------|
| 长文本理解 | 68% | 88% | +20% |
| 多步推理 | 72% | 87% | +15% |
| 风险评估 | 82% | 94% | +12% |

---

## 🔗 相关资源

### 项目内文档
- `README.md` - 项目总体说明
- `COMPLETE_SOLUTION_SUMMARY.md` - 完整方案总结
- `START_TRAINING_HERE.md` - 训练入门指南
- `DATA_GENERATION_SUMMARY.txt` - 数据生成简明总结

### 外部资源
- [GLM-4文档](https://github.com/THUDM/GLM-4)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [GRPO论文](https://arxiv.org/abs/2402.03300)

---

## ✅ 验收清单

在使用数据前，请验证：

- [ ] 已阅读 COMPLEX_TEXT_QUICK_REFERENCE.md
- [ ] 数据文件完整 (CSV + JSONL)
- [ ] 数据字段一致 (17列)
- [ ] 推理链路格式正确 (6步, 以`|`分隔)
- [ ] 风险标签分布正常 (85%-15%)
- [ ] 文本长度在预期范围 (600-800字符)

---

## 📞 快速帮助

### 数据问题
Q: CSV和JSONL有什么区别?  
A: CSV包含所有原始特征+新特征，适合数据分析。JSONL专为模型训练设计。

Q: 能否修改文本内容?  
A: 不建议手动修改。如需调整，请运行generate_complex_kyc_texts.py并修改参数。

### 使用问题
Q: 如何集成到现有训练流程?  
A: 参考 GSPO_TRAINING_GUIDE.md 中的"数据准备"部分。

Q: 支持多语言吗?  
A: 当前版本仅支持中文。多语言版本在规划中。

---

## 🎓 推荐学习路径

### 初级 (1天)
1. 阅读 COMPLEX_TEXT_QUICK_REFERENCE.md (20分钟)
2. 加载数据并探索 (30分钟)
3. 理解数据结构和字段含义 (30分钟)

### 中级 (3天)
1. 深入阅读 COMPLEX_TEXT_FEATURES_SUMMARY.md (1小时)
2. 学习生成算法和代码 (1小时)
3. 运行generate_complex_kyc_texts.py并修改参数 (2小时)
4. 进行数据质量验证 (1小时)

### 高级 (1周)
1. 深入学习 GSPO_TRAINING_GUIDE.md (2小时)
2. 实施模型训练 (4小时)
3. 评估模型性能 (2小时)
4. 进行模型推理和部署 (3小时)
5. 优化和改进 (5小时)

---

## 📋 总结

| 资源 | 类型 | 大小 | 用途 |
|------|------|------|------|
| COMPLEX_TEXT_QUICK_REFERENCE.md | 文档 | 10KB | 快速参考 |
| COMPLEX_TEXT_FEATURES_SUMMARY.md | 文档 | 15KB | 详细说明 |
| GSPO_TRAINING_GUIDE.md | 文档 | 18KB | 实战教程 |
| COMPLEX_TEXT_GENERATION_COMPLETE.md | 报告 | 13KB | 项目总结 |
| kyc_rl_training_dataset_with_complex_text.csv | 数据 | 25MB | 训练数据 |
| kyc_gspo_training_data.jsonl | 数据 | 4MB | GSPO格式 |
| generate_complex_kyc_texts.py | 代码 | 22KB | 生成脚本 |

**总产出**: 93KB文档 + 29MB数据 + 22KB代码

---

**最后更新**: 2026年3月20日  
**版本**: v1.0  
**维护者**: KYC-RL项目组

---

👉 **建议**: 如果这是你第一次接触该功能，请从 **COMPLEX_TEXT_QUICK_REFERENCE.md** 开始！
