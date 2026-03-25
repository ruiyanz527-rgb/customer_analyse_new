"""
生成符合KYC强化学习训练的标准化数据集
基于美团金融LLM强化学习文档中的场景设计
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# ============================================================================
# 定义中文职业类别和收入映射
# ============================================================================
occupations = {
    '技术': {'min_income': 8000, 'max_income': 25000, 'risk': 0.05},
    '金融': {'min_income': 10000, 'max_income': 30000, 'risk': 0.08},
    '教育': {'min_income': 5000, 'max_income': 15000, 'risk': 0.10},
    '销售': {'min_income': 6000, 'max_income': 20000, 'risk': 0.15},
    '制造': {'min_income': 4000, 'max_income': 12000, 'risk': 0.20},
    '医疗': {'min_income': 7000, 'max_income': 22000, 'risk': 0.06},
    '自营': {'min_income': 3000, 'max_income': 35000, 'risk': 0.25},
    '服务': {'min_income': 3000, 'max_income': 10000, 'risk': 0.22},
    '管理': {'min_income': 9000, 'max_income': 28000, 'risk': 0.07},
}

job_titles = {
    '技术': ['技术总监', '工程师', 'CTO', '架构师', '数据科学家'],
    '金融': ['CFO', '投资经理', '风控总监', '分析师', '交易员'],
    '教育': ['教授', '讲师', '教师', '校长', '研究员'],
    '销售': ['销售总监', '销售经理', '销售代表', '区域经理', '大客户经理'],
    '制造': ['生产总监', '工厂经理', '工程师', '技术员', '生产主管'],
    '医疗': ['医生', '医院院长', '护士长', '诊疗主任', '医学博士'],
    '自营': ['企业家', '个体户', '商户', '店主', '自由职业者'],
    '服务': ['餐厅经理', '酒店经理', '服务主管', '前台', '客服'],
    '管理': ['总经理', '副总经理', '部门经理', '项目经理', '人力资源'],
}

cities = ['北京', '上海', '深圳', '杭州', '南京', '武汉', '西安', '成都', '广州', '苏州']
education_levels = ['中专', '高中', '本科', '硕士', '博士']

# ============================================================================
# 辅助函数
# ============================================================================

def generate_kyc_text(user_id, age, occupation, job_title, education, city, income, work_years):
    """生成KYC工单的原始文本"""
    experiences = [
        f"{age-work_years-5}年在{random.choice(cities)}工作过",
        f"拥有{work_years}年{occupation}行业经验",
        f"在{city}定居{random.randint(2,10)}年"
    ]
    
    templates = [
        f"用户{user_id}，{age}岁，{city}市民，学历{education}。现任职位：{job_title}。工作行业：{occupation}。年收入约{income:,.0f}元。{random.choice(experiences)}。",
        f"申请人基本信息：姓名编号{user_id}，出生于{datetime.now().year - age}年，常驻{city}。教育背景：{education}。现职：{job_title}@{occupation}行业。收入水平：{income:,.0f}/年。从业{work_years}年。",
        f"KYC核查结果：用户{user_id}在{occupation}领域任{job_title}一职，月薪约{income/12:,.0f}元。学历{education}，年龄{age}岁，居住地{city}。职业发展稳定，行业内资深人士。",
    ]
    return random.choice(templates)


def generate_risk_reasoning(user_data):
    """生成风险评估的推理过程（模拟人工审核链路）"""
    reasoning_steps = []
    
    # Step 1: 职业分析
    if user_data['occupation'] in ['自营', '服务']:
        reasoning_steps.append(f"[职业风险] {user_data['occupation']}行业风险较高，需重点关注")
    elif user_data['occupation'] in ['金融', '技术']:
        reasoning_steps.append(f"[职业优势] {user_data['occupation']}行业收入稳定，风险较低")
    
    # Step 2: 收入验证
    if user_data['income'] < 5000 and user_data['age'] > 35:
        reasoning_steps.append(f"[收入警告] 年龄{user_data['age']}但收入仅{user_data['income']:,.0f}，低于平均水平")
    
    # Step 3: 行为特征
    if user_data['transaction_frequency'] > 50:
        reasoning_steps.append(f"[交易活跃] 月均交易{user_data['transaction_frequency']}笔，经济活跃度高")
    else:
        reasoning_steps.append(f"[交易稀疏] 月均交易{user_data['transaction_frequency']}笔，可能存在隐性收入")
    
    # Step 4: 综合评分
    if user_data['is_risky'] == 1:
        reasoning_steps.append(f"[高风险] 综合评分{user_data['risk_score']:.2f}，建议拒绝或额度控制")
    else:
        reasoning_steps.append(f"[低风险] 综合评分{user_data['risk_score']:.2f}，可以正常审批")
    
    return reasoning_steps


# ============================================================================
# 主要生成逻辑
# ============================================================================

print("=" * 80)
print("📊 生成符合KYC强化学习的标准化数据集")
print("=" * 80)

n_samples = 10000
records = []

for user_id in range(n_samples):
    age = np.random.randint(20, 65)
    education = np.random.choice(education_levels, p=[0.1, 0.25, 0.45, 0.15, 0.05])
    occupation = np.random.choice(list(occupations.keys()))
    job_title = np.random.choice(job_titles[occupation])
    city = np.random.choice(cities)
    work_years = max(0, age - 20 - np.random.randint(0, 15))
    
    # 收入生成（基于职业和工作年限）
    occ_info = occupations[occupation]
    base_income = np.random.uniform(occ_info['min_income'], occ_info['max_income'])
    income_boost = work_years * 200
    income = min(base_income + income_boost, occ_info['max_income'] * 1.5)
    
    # 芝麻信用分
    sesame_score = max(300, min(850, np.random.normal(600, 100)))
    
    # 交易频率
    transaction_frequency = max(1, int(np.random.normal(30, 15)))
    transaction_amount = np.random.uniform(1000, 50000)
    
    # 风险标签
    risk_base = occupations[occupation]['risk']
    age_risk = 0.02 if age > 50 else (-0.01 if age < 25 else 0)
    income_risk = 0.05 if income < 5000 else (-0.05 if income > 20000 else 0)
    sesame_risk = -0.10 if sesame_score > 700 else (0.10 if sesame_score < 500 else 0)
    
    final_risk_prob = np.clip(risk_base + age_risk + income_risk + sesame_risk, 0, 0.5)
    is_risky = 1 if np.random.random() < final_risk_prob else 0
    
    risk_score = 1.0 - (sesame_score / 850) * (1 - final_risk_prob)
    
    # 生成文本和推理链路
    kyc_text = generate_kyc_text(user_id, age, occupation, job_title, education, city, income, work_years)
    
    user_data = {
        'age': age,
        'occupation': occupation,
        'job_title': job_title,
        'education': education,
        'city': city,
        'income': income,
        'sesame_score': sesame_score,
        'work_years': work_years,
        'transaction_frequency': transaction_frequency,
        'is_risky': is_risky,
        'risk_score': risk_score,
    }
    
    reasoning_steps = generate_risk_reasoning(user_data)
    
    record = {
        'user_id': user_id,
        'age': age,
        'occupation': occupation,
        'job_title': job_title,
        'education': education,
        'city': city,
        'income': income,
        'work_years': work_years,
        'sesame_score': sesame_score,
        'transaction_frequency': transaction_frequency,
        'transaction_amount': transaction_amount,
        'kyc_raw_text': kyc_text,
        'reasoning_chain': '|'.join(reasoning_steps),
        'risk_score': risk_score,
        'is_risky_user': is_risky,
    }
    records.append(record)

df = pd.DataFrame(records)

# 保存完整数据集
output_path = '/Applications/financial LLM/financial_data/kyc_rl_training_dataset.csv'
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"\n✅ 已生成KYC强化学习训练数据集")
print(f"   📁 路径: {output_path}")
print(f"   📊 样本数: {len(df):,}")
print(f"   📈 特征数: {len(df.columns)}")

print("\n" + "=" * 80)
print("📋 数据集结构信息")
print("=" * 80)

print("\n【核心字段说明】")
print("""
1. 结构化特征（用于模型输入）:
   - age: 用户年龄 [20-65]
   - occupation: 职业类别 （9种）
   - job_title: 具体职位 （45种）
   - education: 学历水平 （5个等级）
   - city: 城市 （10个主要城市）
   - income: 年收入 （RMB）
   - work_years: 工作年限
   - sesame_score: 芝麻信用分 [300-850]
   - transaction_frequency: 交易频率 （月均笔数）
   - transaction_amount: 交易金额

2. 非结构化文本（模拟KYC申请文本）:
   - kyc_raw_text: 原始KYC申请文本
   - reasoning_chain: 人工审核推理链路（多步骤标注）

3. 标签与评分（强化学习奖励信号）:
   - risk_score: 风险综合评分 [0-1]
   - is_risky_user: 高风险用户标签 [0/1]
""")

print("\n【数据分布统计】")
print(f"样本总数: {len(df):,}")
print(f"高风险用户占比: {df['is_risky_user'].mean()*100:.2f}%")
print(f"\n职业分布:")
print(df['occupation'].value_counts())
print(f"\n风险评分分布:")
print(df['risk_score'].describe())

print("\n【示例数据（前3条）】")
for idx in range(min(3, len(df))):
    row = df.iloc[idx]
    print(f"\n--- 用户 {row['user_id']} ---")
    print(f"职业: {row['occupation']} | 职位: {row['job_title']}")
    print(f"年龄: {row['age']} | 城市: {row['city']} | 学历: {row['education']}")
    print(f"收入: ¥{row['income']:,.0f} | 芝麻分: {row['sesame_score']:.0f} | 工作年限: {row['work_years']}")
    print(f"KYC文本: {row['kyc_raw_text'][:100]}...")
    print(f"推理链路: {row['reasoning_chain'][:150]}...")
    print(f"风险评分: {row['risk_score']:.3f} | 高风险: {'是' if row['is_risky_user'] else '否'}")

print("\n" + "=" * 80)
print("✨ 数据集生成完毕!")
print("=" * 80)
