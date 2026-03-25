# 🚀 开始使用 - 完整指南

欢迎使用**金融LLM项目**！本文档将帮助您快速上手。

---

## ⚡ 5分钟快速开始

### 第1步：安装依赖
```bash
cd /Applications/financial\ LLM
pip install -r requirements.txt
```

### 第2步：运行快速开始脚本
```bash
python3 quick_start.py
```

**预期结果**：
- ✅ 生成50,000条支付宝数据
- ✅ 生成50,000条美团数据
- ✅ 生成50,000条融合数据
- ✅ 显示数据统计信息
- ✅ 执行完整预处理流程

### 第3步：验证结果
```bash
ls -lh financial_data/
```

你应该看到3个CSV文件：
- `alipay_users.csv` (~10MB)
- `meituan_users.csv` (~10MB)
- `combined_users.csv` (~8MB)

---

## 📚 完整学习路径

### 初级 (适合快速验证)

**目标**: 理解项目结构和数据格式

```
1. 阅读 README.md              (5分钟)
   ↓
2. 运行 quick_start.py          (2分钟)
   ↓
3. 查看生成的数据               (2分钟)
   ↓
4. 总时间: ~10分钟 ✅
```

### 中级 (适合数据处理)

**目标**: 学会加载、预处理数据

```python
# 代码示例
from data_loader import FinancialDataLoader, DataPreprocessor

# 1. 加载数据
loader = FinancialDataLoader()
alipay_data = loader.load_alipay()
print(f"加载了 {len(alipay_data)} 条数据")

# 2. 预处理
preprocessor = DataPreprocessor()
processed = preprocessor.pipeline(alipay_data)
print(f"预处理完成: {processed.shape}")

# 3. 保存处理后的数据
processed.to_csv('alipay_processed.csv', index=False)
```

**学习资源**:
- 📄 README.md - 数据说明
- 📄 PROJECT_STRUCTURE.md - 理解 data_loader.py
- 🐍 data_loader.py - 源代码

**预计时间**: 30分钟

### 高级 (适合模型训练)

**目标**: 获取真实数据并训练模型

```python
# 第1步: 获取真实数据
# 参考 dataset_acquisition_guide.md

# 第2步: 加载和预处理
from data_loader import FinancialDataLoader, DataPreprocessor, DataSplitter

df = FinancialDataLoader.load_custom_csv('real_data.csv')
preprocessor = DataPreprocessor()
processed = preprocessor.pipeline(df)

# 第3步: 数据分割
splitter = DataSplitter()
train_df, test_df = splitter.train_test_split(processed, test_size=0.2)

# 第4步: 模型训练
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

X_train = train_df.drop('is_default', axis=1)
y_train = train_df['is_default']

model = XGBClassifier()
model.fit(X_train, y_train)

# 第5步: 模型评估
X_test = test_df.drop('is_default', axis=1)
y_test = test_df['is_default']

score = model.score(X_test, y_test)
print(f"模型准确率: {score:.2%}")
```

**学习资源**:
- 📄 dataset_acquisition_guide.md - 数据获取
- 📄 INSTALL.md - 环境配置
- 🐍 data_generator.py - 生成更多数据
- 🐍 tianchi_data_downloader.py - 下载真实数据

**预计时间**: 2-3小时

---

## 🎯 常见任务

### 任务1：探索数据特征
```python
import pandas as pd

df = pd.read_csv('financial_data/alipay_users.csv')

# 基本统计
print(df.describe())

# 缺失值检查
print(df.isnull().sum())

# 数据类型
print(df.dtypes)

# 可视化
import matplotlib.pyplot as plt
df['is_default'].value_counts().plot(kind='bar')
plt.show()
```

### 任务2：自定义预处理
```python
from data_loader import DataPreprocessor

preprocessor = DataPreprocessor()

# 自定义预处理步骤
steps = {
    'missing_values': {'strategy': 'mean'},
    'outliers': {'method': 'iqr'},
    'encode': {'columns': ['occupation', 'education']},
    'normalize': {'method': 'zscore'},
}

processed = preprocessor.pipeline(df, steps=steps)
```

### 任务3：创建训练集和验证集
```python
from data_loader import DataSplitter

splitter = DataSplitter()

# 简单分割
train_df, test_df = splitter.train_test_split(df, test_size=0.2)

# K折交叉验证
folds = splitter.k_fold_split(df, n_splits=5)

for i, (train, val) in enumerate(folds):
    print(f"Fold {i+1}: Train={len(train)}, Val={len(val)}")
```

### 任务4：导入天池数据
```bash
python3 tianchi_data_downloader.py
```

然后选择选项3 (导入CSV文件)

### 任务5：生成更多样本
```python
from data_generator import AlipayDataGenerator

gen = AlipayDataGenerator()
data = gen.generate(n_samples=100000)  # 生成100,000条
data.to_csv('alipay_large.csv', index=False)
```

---

## 📁 项目文件导航

### 📖 文档
| 文件 | 用途 | 阅读时间 |
|-----|------|--------|
| README.md | 项目概览 | 5分钟 |
| INSTALL.md | 安装指南 | 10分钟 |
| GETTING_STARTED.md | 本文件 | 10分钟 |
| skills.md | 业务分析 | 15分钟 |
| dataset_acquisition_guide.md | 数据获取 | 20分钟 |
| PROJECT_STRUCTURE.md | 项目结构 | 15分钟 |

### 🐍 代码
| 文件 | 功能 | 难度 |
|-----|------|------|
| quick_start.py | 一键快速开始 | ⭐ |
| data_generator.py | 生成合成数据 | ⭐⭐ |
| data_loader.py | 加载和预处理 | ⭐⭐ |
| tianchi_data_downloader.py | 下载真实数据 | ⭐ |

### 💾 数据
| 文件 | 行数 | 用途 |
|-----|------|------|
| alipay_users.csv | 50,000 | 支付宝数据 |
| meituan_users.csv | 50,000 | 美团数据 |
| combined_users.csv | 50,000 | 融合数据 |

---

## 🎓 核心概念

### 什么是KYC?
**KYC** = Know Your Customer (了解你的客户)

在金融服务中，KYC是指：
- 收集和验证用户身份信息
- 评估客户的金融风险
- 进行合规性检查

**本项目的KYC维度**:
- 基础信息: 年龄、职业、教育程度、地址
- 行为数据: 交易频率、消费金额、活跃度
- 消费画像: 消费场景、消费偏好、支出结构
- 负债状况: 贷款记录、信用评分、逾期情况

### 三大业务痛点

#### 痛点1: 低频、稀疏与长尾问题
- **问题**: 新客数据不完整，关键特征缺失
- **解决**: 使用包含长尾样本的数据集 (如LendingClub)
- **数据集**: UCIC Credit, Fraud Detection

#### 痛点2: 非结构化数据处理
- **问题**: 文本中的关键信息难以提取
- **解决**: 使用LLM进行文本理解
- **数据集**: LendingClub (包含职位描述、申请理由)

#### 痛点3: 复杂推理能力缺失
- **问题**: 多源数据综合推理困难
- **解决**: 融合多平台数据，创建关联特征
- **数据集**: 支付宝 + 美团融合数据

---

## 💡 最佳实践

### ✅ 推荐做法

1. **先用生成数据验证流程**
   ```python
   # 快速验证
   python3 quick_start.py
   ```

2. **然后获取真实数据**
   ```bash
   python3 tianchi_data_downloader.py
   ```

3. **最后规模化训练**
   ```python
   # 使用更多样本进行训练
   ```

### ❌ 常见错误

1. **不进行数据预处理就训练模型**
   - ❌ 错误: `model.fit(df, y)`
   - ✅ 正确: `processed = preprocessor.pipeline(df)`

2. **忽视数据不平衡**
   - ❌ 错误: 忽视目标变量的不平衡
   - ✅ 正确: 使用过采样/欠采样或加权

3. **没有进行特征工程**
   - ❌ 错误: 直接使用原始特征
   - ✅ 正确: 创建衍生特征

4. **未进行交叉验证**
   - ❌ 错误: 只用train/test分割
   - ✅ 正确: 使用K折交叉验证

---

## 🔗 相关链接

### 官方文档
- [Pandas 文档](https://pandas.pydata.org/docs/)
- [Scikit-learn 文档](https://scikit-learn.org/stable/)
- [XGBoost 文档](https://xgboost.readthedocs.io/)

### 数据源
- [阿里天池](https://tianchi.aliyun.com/)
- [Kaggle](https://www.kaggle.com/datasets)
- [UCI Machine Learning](https://archive.ics.uci.edu/)

### 课程资源
- [吴恩达机器学习课程](https://www.coursera.org/learn/machine-learning)
- [Fast.ai 深度学习](https://www.fast.ai/)

---

## ❓ 常见问题

**Q: 生成的数据是真实的吗?**
A: 不是。这些都是合成数据，用于演示和测试。真实数据请从天池、Kaggle等渠道获取。

**Q: 可以用生成的数据训练生产模型吗?**
A: 可以作为POC (概念验证)，但生产模型应使用真实数据。

**Q: 如何处理数据不平衡?**
A: 
```python
from sklearn.utils import class_weight

# 计算类别权重
weights = class_weight.compute_class_weight('balanced', 
                                             classes=np.unique(y), 
                                             y=y)
```

**Q: 需要多少样本才能训练?**
A: 最少10,000条样本。推荐使用50,000-100,000条以上。

**Q: 支持中文吗?**
A: 支持。所有代码和数据都支持中文处理。使用 `encoding='utf-8-sig'` 读写CSV文件。

---

## 📞 获取帮助

遇到问题？查看以下资源：

1. **快速问题** → 查看本文件 (GETTING_STARTED.md)
2. **安装问题** → 查看 INSTALL.md
3. **数据问题** → 查看 dataset_acquisition_guide.md
4. **代码问题** → 查看 PROJECT_STRUCTURE.md
5. **业务问题** → 查看 skills.md

---

## 🎉 下一步

现在你已经准备好了！

### 立即开始
```bash
python3 quick_start.py
```

### 或者选择一条学习路径
- 🟢 **初级**: 5分钟快速开始
- 🟡 **中级**: 30分钟数据处理
- 🔴 **高级**: 2小时完整流程

---

**祝您学习愉快！** 🚀

*最后更新: 2026年3月*

