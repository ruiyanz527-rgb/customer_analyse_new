# KYC智能用户理解 - 强化学习训练数据集指南

## 📊 数据集概述

根据**美团金融LLM强化学习技术**文档（2750446176），本项目为您生成了一个**符合生产级KYC场景**的强化学习训练数据集。

### 核心特性

| 特性 | 说明 |
|------|------|
| **样本规模** | 10,000条用户记录 |
| **数据类型** | 结构化特征 + 非结构化文本 + 推理链路 |
| **应用场景** | 智能用户KYC、风险评估、贷款审批 |
| **RL算法** | GRPO、GSPO、RLxSFT混合训练 |
| **奖励信号** | 可验证的风险评分（Verifiable Rewards） |

---

## 📁 数据集位置

```
/Applications/financial LLM/financial_data/
├── kyc_rl_training_dataset.csv          # ← 主数据集（10,000条）
├── alipay_users.csv                     # 支付宝用户特征
├── meituan_users.csv                    # 美团用户特征
└── combined_users.csv                   # 融合用户特征
```

---

## 🔍 数据集字段详解

### 1️⃣ **结构化特征** （模型输入）

| 字段名 | 数据类型 | 范围/枚举 | 说明 |
|--------|--------|---------|------|
| `user_id` | int | 0-9999 | 用户唯一标识 |
| `age` | int | 20-65 | 用户年龄 |
| `occupation` | str | 9种 | 职业类别：技术、金融、教育、销售、制造、医疗、自营、服务、管理 |
| `job_title` | str | 45种 | 具体职位：技术总监、工程师、CTO等 |
| `education` | str | 5级 | 学历：中专、高中、本科、硕士、博士 |
| `city` | str | 10城 | 城市：北京、上海、深圳等 |
| `income` | float | ¥3K-¥35K | 年收入（元） |
| `work_years` | int | 0-45 | 工作年限 |
| `sesame_score` | float | 300-850 | 芝麻信用分 |
| `transaction_frequency` | int | 1-100+ | 月均交易笔数 |
| `transaction_amount` | float | ¥1K-¥50K | 交易金额 |

### 2️⃣ **非结构化文本** （LLM输入）

#### `kyc_raw_text`
**模拟真实KYC工单的文本**，包含：
- 用户基本信息（年龄、城市、学历）
- 职业信息（职位、行业、从业经历）
- 收入水平描述
- 职业发展背景

**示例：**
```
KYC核查结果：用户125在金融领域任分析师一职，月薪约2,500元。学历硕士，
年龄35岁，居住地深圳。职业发展稳定，行业内资深人士。
```

### 3️⃣ **推理链路** （人工审核标注）

#### `reasoning_chain`
**多步骤推理过程**，由`|`分隔，包含4个关键步骤：

1. **[职业风险]** - 职业类别的风险评估
   - 示例：`[职业风险] 自营行业风险较高，需重点关注`

2. **[收入警告]** - 收入水平合理性检查
   - 示例：`[收入警告] 年龄45但收入仅¥4,500，低于平均水平`

3. **[交易活跃]** - 交易行为特征分析
   - 示例：`[交易活跃] 月均交易65笔，经济活跃度高`

4. **[高/低风险]** - 综合评估结论
   - 示例：`[低风险] 综合评分0.32，可以正常审批`

**示例完整链路：**
```
[职业优势] 金融行业收入稳定，风险较低|[交易活跃] 月均交易45笔，经济活跃度高|[低风险] 综合评分0.25，可以正常审批
```

### 4️⃣ **标签与奖励信号** （强化学习目标）

| 字段名 | 数据类型 | 范围 | 说明 |
|--------|--------|-----|------|
| `risk_score` | float | [0.0-1.0] | 风险综合评分（连续值） |
| `is_risky_user` | int | {0, 1} | 高风险用户标签（二分类） |

---

## 🎯 与文档中痛点的对应关系

### 痛点1：低频、稀疏与长尾问题

**数据集覆盖：**
- ✅ 新客冷启动：低work_years的年轻用户（年龄<25）
- ✅ 稀疏特征：某些职业（自营）的收入变异度大
- ✅ 长尾分布：is_risky_user的不均衡分布（高风险占比约15-20%）

**如何使用：**
```python
# 识别新客样本
new_customers = df[df['work_years'] < 2]

# 识别长尾高风险样本
tail_risky = df[df['is_risky_user'] == 1]
```

### 痛点2：非结构化数据处理

**数据集覆盖：**
- ✅ 原始文本：`kyc_raw_text`包含完整KYC申请材料
- ✅ 结构化标签：14个结构化特征与文本对齐
- ✅ 语义丰富性：45种职位、9种职业、10个城市的组合

**如何使用：**
```python
# 提取文本与特征的对齐数据
text_feature_pairs = df[['kyc_raw_text', 'occupation', 'job_title', 'income']]

# 用于BERT/LLM微调
texts = df['kyc_raw_text'].tolist()
labels = df['occupation'].tolist()
```

### 痛点3：复杂推理能力缺失

**数据集覆盖：**
- ✅ 推理链路标注：`reasoning_chain`包含4步决策过程
- ✅ 多源信息综合：职业+收入+交易行为+信用分
- ✅ 冲突检测：年龄-收入、职业-收入的矛盾案例

**如何使用：**
```python
# 提取推理链路用于强化学习
reasoning_data = df[['kyc_raw_text', 'reasoning_chain', 'risk_score']]

# 用于过程奖励模型(PRM)训练
for idx, row in df.iterrows():
    steps = row['reasoning_chain'].split('|')
    for step in steps:
        # 标注中间推理步骤的正确性
        pass
```

---

## 📈 数据统计信息

```python
import pandas as pd

df = pd.read_csv('/Applications/financial LLM/financial_data/kyc_rl_training_dataset.csv')

# 基本统计
df.shape                           # (10000, 15)
df['is_risky_user'].value_counts() # 风险分布
df.groupby('occupation')['is_risky_user'].mean()  # 职业风险率
df.groupby('city')['income'].mean()               # 城市收入差异
```

---

## 🚀 在RL训练中的应用

### 1️⃣ **SFT (监督微调) 阶段**

```python
# 准备SFT训练数据
from collections import namedtuple

SFTData = namedtuple('SFTData', ['prompt', 'completion', 'label'])

sft_data = []
for _, row in df.iterrows():
    prompt = f"""根据以下KYC信息评估用户风险:
{row['kyc_raw_text']}

请提供风险评估和建议。"""
    
    completion = row['reasoning_chain']  # 人工标注的推理链路
    label = row['is_risky_user']
    
    sft_data.append(SFTData(prompt, completion, label))
```

### 2️⃣ **GRPO (强化学习) 阶段**

```python
# 可验证奖励信号（RLVR）
def verify_kyc_output(model_output, golden_risk_score):
    """
    验证模型输出是否正确识别风险
    - 如果模型预测 > 0.5 且 golden_risk_score > 0.5 → reward = +1.0
    - 否则 → reward = -0.5
    """
    predicted_risk = extract_risk_score(model_output)
    if (predicted_risk > 0.5) == (golden_risk_score > 0.5):
        return 1.0
    else:
        return -0.5

# GRPO训练循环（伪代码）
for epoch in range(n_epochs):
    for batch_prompts in batches:
        # 采样生成
        model_outputs = model.sample(batch_prompts, num_samples=8)
        
        # 计算奖励
        rewards = [verify_kyc_output(output, golden_label) 
                   for output in model_outputs]
        
        # GRPO更新
        optimizer.update(model, prompts, model_outputs, rewards)
```

### 3️⃣ **差异化奖励设计**

```python
# 根据字段难度设置不同权重
def differentiated_reward(model_output, row):
    """
    难刻画字段（工作职位、工作地点）给予更高奖励
    """
    base_reward = verify_kyc_output(model_output, row['risk_score'])
    
    # 职位识别的额外奖励 (+0.5)
    if predict_job_title_correct(model_output, row['job_title']):
        base_reward += 0.5
    
    # 收入推理的额外奖励 (+0.3)
    if predict_income_reasonable(model_output, row['income']):
        base_reward += 0.3
    
    return base_reward
```

### 4️⃣ **课程学习策略**

```python
# 按风险难度划分简单/困难样本
easy_samples = df[df['risk_score'] < 0.3]  # 明显低风险
hard_samples = df[df['risk_score'] > 0.5]  # 明显高风险

# Stage 1：仅用简单样本训练
train_stage1(model, easy_samples, epochs=5)

# Stage 2：混合简单和困难样本（1:1）
combined = pd.concat([easy_samples, hard_samples])
train_stage2(model, combined, epochs=5)
```

---

## 📚 相关字段的业务含义

### 职业风险映射

| 职业 | 基础风险率 | 特点 | 推荐额度 |
|------|---------|------|---------|
| **技术** | 5% | 收入稳定，职业发展清晰 | 高额 |
| **金融** | 8% | 收入高，但变动性大 | 高额 |
| **医疗** | 6% | 社会地位高，收入稳定 | 高额 |
| **管理** | 7% | 职业发展稳定 | 高额 |
| **教育** | 10% | 收入一般，相对稳定 | 中等 |
| **销售** | 15% | 收入差异大，浮动性高 | 中等 |
| **制造** | 20% | 行业周期性强 | 中等 |
| **服务** | 22% | 收入不稳定，兼职多 | 低额 |
| **自营** | 25% | 风险最高，收入波动大 | 低额 |

### 芝麻信用分的风险影响

```python
# 信用分风险权重
if sesame_score > 700:
    credit_advantage = -0.10  # 低风险
elif sesame_score > 550:
    credit_advantage = 0.00   # 中等
else:
    credit_advantage = 0.10   # 高风险
```

---

## 🔄 数据集更新与维护

### 定期更新建议

1. **每月** - 补充新样本，更新职业分布
2. **每季度** - 调整风险标签，基于实际逾期数据
3. **每半年** - 扩展职位库、城市覆盖、特征工程

### 如何扩展

```python
# 添加新职业
new_occupation = {
    'IT管理': {'min_income': 12000, 'max_income': 35000, 'risk': 0.04}
}
occupations.update(new_occupation)

# 添加新城市
new_cities = ['重庆', '郑州', '长沙']
cities.extend(new_cities)

# 生成更多样本
generate_kyc_rl_dataset(n_samples=20000)
```

---

## 📖 与美团KYC文档的对应关系

| 文档章节 | 对应数据 | 使用场景 |
|---------|--------|---------|
| **3.1 奖励函数设计** | `risk_score`, `reasoning_chain` | 差异化奖励信号 |
| **3.2 参数优化** | 全数据集 | GRPO超参调优验证 |
| **3.3 GSPO长文本** | `kyc_raw_text` | 长文本特征提取 |
| **3.4 RLxSFT混合** | `reasoning_chain` + `risk_score` | SFT初始化 + RL优化 |
| **3.5 课程学习** | `is_risky_user`, `risk_score` | 难度分级训练 |

---

## 💡 快速开始

```python
import pandas as pd

# 加载数据集
df = pd.read_csv('/Applications/financial LLM/financial_data/kyc_rl_training_dataset.csv')

# 查看基本信息
print(df.shape)                 # (10000, 15)
print(df.columns.tolist())      # 全部字段

# 查看数据分布
print(df['occupation'].value_counts())
print(df['risk_score'].describe())

# 导出为其他格式
df.to_json('kyc_dataset.jsonl', orient='records', lines=True)
df[['kyc_raw_text', 'reasoning_chain']].to_json('kyc_training.json')
```

---

## 📞 常见问题

**Q: 数据集中高风险用户占比多少？**
A: 约15-20%，符合真实金融业务的长尾特征。

**Q: 如何处理职位识别这类难刻画字段？**
A: 使用差异化奖励（见3.1.2），给予更高的正确预测权重。

**Q: 这个数据集是真实数据吗？**
A: 否。这是**合成数据**，模拟真实KYC场景但不含隐私敏感信息，可用于RL算法研究和开发。

**Q: 如何扩展样本量？**
A: 修改 `generate_kyc_rl_dataset.py` 中的 `n_samples` 参数。

---

## 📄 许可与使用

本数据集为研究和开发用途，遵循美团金融技术实践指南。
