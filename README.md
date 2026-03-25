# 金融LLM训练项目

基于大模型的KYC（了解你的客户）和风险管理系统

## 📁 项目结构

```
financial LLM/
├── README.md                      # 项目说明（本文件）
├── skills.md                      # 技能和数据集分析
├── dataset_acquisition_guide.md   # 数据获取详细指南
├── data_generator.py              # 合成数据生成器
├── data_loader.py                 # 数据加载和预处理工具
└── financial_data/                # 生成的数据目录
    ├── alipay_users.csv          # 支付宝用户数据
    ├── meituan_users.csv         # 美团用户数据
    └── combined_users.csv        # 融合用户数据
```

## 🚀 快速开始

### 第1步：生成测试数据

```bash
python data_generator.py
```

这将生成：
- `alipay_users.csv` - 50,000条支付宝用户数据
- `meituan_users.csv` - 50,000条美团用户数据  
- `combined_users.csv` - 50,000条融合用户数据

### 第2步：加载和预处理数据

```python
from data_loader import FinancialDataLoader, DataPreprocessor

# 加载数据
loader = FinancialDataLoader()
alipay_data = loader.load_alipay()

# 预处理
preprocessor = DataPreprocessor()
processed_data = preprocessor.pipeline(alipay_data)

# 查看结果
print(processed_data.head())
print(processed_data.info())
```

### 第3步：划分训练/测试集

```python
from data_loader import DataSplitter

splitter = DataSplitter()
train_df, test_df = splitter.train_test_split(processed_data, test_size=0.2)

print(f"训练集: {train_df.shape}")
print(f"测试集: {test_df.shape}")
```

## 📊 数据说明

### 支付宝用户数据特征

**基础信息维度**：
- `user_id` - 用户ID
- `age` - 年龄
- `occupation` - 职业
- `education` - 教育程度
- `city`, `province` - 城市/省份

**账户特征**：
- `account_age_days` - 账户年龄
- `kyc_verified` - KYC认证状态
- `sesame_score` - 芝麻信用分

**交易特征**：
- `total_transaction_amount` - 总交易金额
- `total_transaction_count` - 总交易笔数
- `avg_transaction_amount` - 平均交易金额

**借贷特征**：
- `has_loan` - 是否有贷款
- `active_loan_amount` - 活跃贷款余额
- `loan_repay_record` - 贷款还款记录

**理财特征**：
- `has_yebao` - 是否开通余额宝
- `yebao_balance` - 余额宝余额

**风险标签**：
- `is_default` - 是否违约（目标变量）

### 美团用户数据特征

**基础信息维度**：
- `user_id` - 用户ID
- `age` - 年龄
- `city` - 城市
- `registration_days` - 注册天数

**订单特征**：
- `total_orders` - 总订单数
- `avg_order_amount` - 平均订单金额
- `total_spending` - 总消费金额
- `monthly_order_frequency` - 月均订单频次

**消费偏好**：
- `favorite_cuisine` - 喜欢的菜系
- `breakfast_consumer`, `lunch_consumer`, `dinner_consumer`, `late_night_consumer` - 消费时段

**支付行为**：
- `payment_on_time_rate` - 按时支付率
- `meipay_usage_ratio` - 美团支付使用比例
- `complaint_count` - 投诉次数

**金融产品**：
- `has_credit_product` - 是否有信用产品
- `meituan_loan_available` - 美团贷可用额度
- `meituan_loan_used` - 美团贷已用额度

**信用等级**：
- `credit_level` - 信用等级（青铜/白银/黄金/铂金）
- `is_high_risk` - 是否高风险用户

### 融合特征

融合数据同时包含支付宝和美团的关键特征，以及综合衍生特征：
- `total_platform_spending` - 总平台消费
- `platform_diversity` - 平台多样性
- `financial_activity_score` - 金融活跃度评分
- `is_risky_user` - 综合风险标签

## 🔧 数据预处理管道

### 支持的预处理操作

1. **缺失值处理** - mean/median/drop/forward_fill
2. **异常值移除** - IQR/zscore方法
3. **分类编码** - 支持职业、学历、信用等级等
4. **特征归一化** - MinMax/ZScore/Log变换
5. **衍生特征创建** - 自动生成金融比率指标

### 使用示例

```python
from data_loader import DataPreprocessor

preprocessor = DataPreprocessor()

# 自定义预处理步骤
steps = {
    'missing_values': {'strategy': 'mean'},
    'outliers': {'method': 'iqr'},
    'encode': {},
    'derive': {},
    'normalize': {'method': 'zscore'}  # 改用ZScore
}

processed_data = preprocessor.pipeline(df, steps=steps)
```

## 📖 获取真实数据

详细的数据获取指南请参考 `dataset_acquisition_guide.md`，包括：

- ✅ **阿里天池平台** - 真实脱敏数据，免费下载
- ✅ **Kaggle数据集** - 业界标准数据集
- ✅ **合成数据生成** - 快速验证模型
- ✅ **企业合作渠道** - 生产级数据

## 🎯 核心业务痛点解决

### 痛点1：低频、稀疏与长尾问题
- ✓ 生成包含新客、冷启动场景的数据
- ✓ 支持长尾分布样本
- ✓ 多时间跨度纵向数据

### 痛点2：非结构化数据处理
- ✓ 职业、地址等自由文本字段
- ✓ 原始特征与结构化标签对齐
- ✓ 领域术语处理能力

### 痛点3：复杂推理能力缺失
- ✓ 多源异构信息综合（支付宝+美团）
- ✓ 职业-收入-资产关联映射
- ✓ 跨维度的推理链路

## 📚 文件说明

### skills.md
包含：
- KYC业务需求分析
- 5大数据集推荐（UCIC, LendingClub, Fraud Detection等）
- 数据集对标矩阵
- 3种组合方案

### dataset_acquisition_guide.md
包含：
- 阿里天池平台数据获取
- Kaggle数据集获取
- 学术/研究数据渠道
- 合成数据生成代码
- 企业合作联系方式

### data_generator.py
包含类：
- `AlipayDataGenerator` - 支付宝数据生成器
- `MeituanDataGenerator` - 美团数据生成器
- `CombinedDataGenerator` - 融合数据生成器

### data_loader.py
包含类：
- `FinancialDataLoader` - 数据加载
- `DataPreprocessor` - 数据预处理
- `DataSplitter` - 数据分割

## 🔍 数据质量检查

生成的数据集自动包含：
- ✓ 合理的特征分布
- ✓ 真实的相关性结构
- ✓ 适当的缺失值和异常值
- ✓ 平衡的目标标签分布（可配置）

## 📈 下一步

1. **探索性数据分析 (EDA)**
   ```bash
   jupyter notebook
   ```

2. **特征工程**
   - 基于业务逻辑创建交互特征
   - 选择最相关的特征子集

3. **模型训练**
   - 使用XGBoost/LightGBM进行风险评分
   - 使用Transformer进行序列推理

4. **模型优化**
   - 超参数调优
   - 交叉验证评估
   - 特征重要性分析

## 📞 支持

有任何问题，请参考：
- 📄 `README.md` - 快速开始
- 📄 `skills.md` - 业务背景
- 📄 `dataset_acquisition_guide.md` - 数据获取

## 📝 许可证

本项目仅供学习和研究使用。

---

**最后更新**: 2026年3月

