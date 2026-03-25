# 安装和配置指南

## 环境要求

- Python 3.7+
- 推荐: Python 3.9 或 3.10
- macOS / Linux / Windows

## 快速安装

### 1. 克隆或下载项目

```bash
cd /Applications/financial\ LLM
```

### 2. 创建虚拟环境（推荐）

**使用 venv**:
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate  # Windows
```

**使用 conda**:
```bash
conda create -n financial-llm python=3.9
conda activate financial-llm
```

### 3. 安装依赖

```bash
# 安装所有依赖
pip install -r requirements.txt

# 或分步安装（如果上面的方式有问题）
pip install pandas numpy scipy scikit-learn matplotlib seaborn plotly

# 仅安装基础版本（轻量级）
pip install pandas numpy scikit-learn
```

### 4. 验证安装

```bash
python3 -c "import pandas; import numpy; import sklearn; print('✅ 基础依赖安装成功')"
```

## 快速运行

### 运行快速开始脚本

```bash
python3 quick_start.py
```

这将：
1. ✅ 生成合成数据集
2. ✅ 加载数据
3. ✅ 执行预处理
4. ✅ 显示数据统计
5. ✅ 进行数据分割

**预期输出**:
```
======================================================================
🚀 金融LLM项目 - 快速开始脚本
======================================================================

📊 第1步：生成合成数据...
----------------------------------------------------------------------
🔄 正在生成数据...
  - 生成支付宝数据...
    ✅ 保存到: ./financial_data/alipay_users.csv (50000 行)
  ...

✅ 快速开始完成！所有数据已准备就绪
======================================================================
```

## 按需安装

### 仅用于数据处理

```bash
pip install pandas numpy scikit-learn
python3 data_generator.py
```

### 用于数据分析和可视化

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly jupyter
python3 -m jupyter notebook
```

### 用于模型训练

```bash
pip install pandas numpy scikit-learn xgboost lightgbm
```

### 用于LLM和NLP

```bash
pip install torch transformers
# 或使用CPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 获取阿里/美团数据

### 方案1：从天池平台获取（推荐）

#### 步骤1：注册天池账号
- 访问: https://tianchi.aliyun.com/
- 点击"注册"创建账号
- 使用淘宝账号或企业账号登录

#### 步骤2：查找数据集

搜索关键词：
- `风控` - 风险管理相关数据
- `信用预测` - 信用评分数据
- `O2O` - 用户消费数据
- `蚂蚁` - 蚂蚁金服相关

#### 步骤3：下载数据

- 找到感兴趣的数据集
- 点击"数据集"→"下载"
- 选择"个人使用"（免费）
- 下载CSV文件到 `./financial_data/` 目录

#### 步骤4：在脚本中使用

```python
from data_loader import FinancialDataLoader

# 加载自定义数据
df = FinancialDataLoader.load_custom_csv('./financial_data/alipay_real.csv')

# 预处理
from data_loader import DataPreprocessor
preprocessor = DataPreprocessor()
processed = preprocessor.pipeline(df)
```

### 方案2：从Kaggle获取

#### 步骤1：安装Kaggle CLI

```bash
pip install kaggle
```

#### 步骤2：配置Kaggle API

- 访问: https://www.kaggle.com/settings/account
- 点击"Create New API Token"
- 下载 `kaggle.json`
- 将文件复制到 `~/.kaggle/kaggle.json`
- 修改权限: `chmod 600 ~/.kaggle/kaggle.json`

#### 步骤3：搜索和下载数据集

```bash
# 列出所有金融相关数据集
kaggle datasets list -s "credit"

# 下载特定数据集
kaggle datasets download -d mlg-ulb/creditcardfraud -p ./financial_data

# 解压
unzip financial_data/creditcardfraud.zip -d financial_data/
```

**常用数据集**:
```bash
# 信用卡欺诈检测
kaggle datasets download -d mlg-ulb/creditcardfraud

# 贷款违约预测
kaggle datasets download -d wordsforthewise/lending-club

# 蚂蚁金服风控
kaggle datasets download -d thedevastator/ant-financial-credit-risk-prediction
```

### 方案3：生成合成数据（立即可用）

```bash
# 生成50,000条合成数据（默认）
python3 data_generator.py

# 或在Python中定制
from data_generator import AlipayDataGenerator

gen = AlipayDataGenerator()
data = gen.generate(n_samples=100000)  # 生成100,000条
data.to_csv('./financial_data/custom_data.csv', index=False)
```

## 常见问题

### Q: 运行时提示缺少模块？
**A**: 运行以下命令安装缺失的模块
```bash
pip install [module_name]
```

### Q: 如何使用GPU加速？
**A**: 安装GPU版本的依赖
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或使用conda
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Q: 生成的数据在哪里？
**A**: 默认在 `./financial_data/` 目录下
- `alipay_users.csv` - 支付宝数据
- `meituan_users.csv` - 美团数据
- `combined_users.csv` - 融合数据

### Q: 如何修改生成数据的样本量？
**A**: 编辑 `data_generator.py` 中的最后一行
```python
if __name__ == '__main__':
    save_data_to_csv(output_dir='./financial_data')  # 默认50,000
    # 改为:
    # alipay_gen = AlipayDataGenerator()
    # data = alipay_gen.generate(n_samples=100000)  # 生成100,000条
```

### Q: 如何只预处理特定列？
**A**: 使用 `DataPreprocessor` 的自定义参数
```python
preprocessor = DataPreprocessor()
steps = {
    'missing_values': {'strategy': 'mean'},
    'encode': {'columns': ['occupation', 'education']},
    'normalize': {'method': 'zscore'}
}
processed = preprocessor.pipeline(df, steps=steps)
```

### Q: 如何处理中文字段？
**A**: 使用正确的编码
```python
import pandas as pd

# 读取
df = pd.read_csv('data.csv', encoding='utf-8-sig')

# 写入
df.to_csv('output.csv', encoding='utf-8-sig', index=False)
```

## 性能优化

### 处理大型数据集

```python
import pandas as pd

# 分块读取
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    processed = preprocessor.pipeline(chunk)
    chunks.append(processed)

result = pd.concat(chunks, ignore_index=True)
```

### 并行处理

```python
from multiprocessing import Pool

def process_batch(batch):
    return preprocessor.pipeline(batch)

# 使用多进程加速
with Pool(4) as p:  # 4个进程
    results = p.map(process_batch, batches)
```

## 下一步

1. **阅读文档**
   - `README.md` - 项目概览
   - `skills.md` - 技能分析
   - `dataset_acquisition_guide.md` - 详细获取指南

2. **运行示例**
   ```bash
   python3 quick_start.py
   ```

3. **探索数据**
   ```bash
   python3 -m jupyter notebook
   ```

4. **开始训练模型**
   - 使用生成或下载的数据
   - 参考 `data_loader.py` 进行预处理
   - 使用 XGBoost/LightGBM 训练

## 技术支持

有任何问题，请参考：
- 📄 本文件 (INSTALL.md)
- 📄 README.md
- 📄 dataset_acquisition_guide.md

---

**祝您使用愉快！** 🎉

