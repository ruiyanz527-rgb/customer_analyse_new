import pandas as pd
import numpy as np

# 加载原始数据和扩展数据
original_df = pd.read_csv('financial_data/kyc_rl_training_dataset.csv')
expanded_df = pd.read_csv('financial_data/kyc_rl_training_dataset_expanded_50k.csv')

print("="*80)
print("📊 数据扩展质量验证报告")
print("="*80)

print("\n【数据规模对比】")
print(f"原始数据:  {len(original_df):>6,} 条")
print(f"扩展数据:  {len(expanded_df):>6,} 条")
print(f"增长倍数:  {len(expanded_df)/len(original_df):>6.1f}x")

print("\n【关键特征分布对比】")

# 年龄分布
print(f"\n年龄(age):")
print(f"  原始 - 平均: {original_df['age'].mean():.1f} | 标准差: {original_df['age'].std():.1f} | 范围: [{original_df['age'].min()}-{original_df['age'].max()}]")
print(f"  扩展 - 平均: {expanded_df['age'].mean():.1f} | 标准差: {expanded_df['age'].std():.1f} | 范围: [{expanded_df['age'].min()}-{expanded_df['age'].max()}]")

# 收入分布
print(f"\n收入(income):")
print(f"  原始 - 平均: ¥{original_df['income'].mean():>10,.0f} | 标准差: ¥{original_df['income'].std():>10,.0f}")
print(f"  扩展 - 平均: ¥{expanded_df['income'].mean():>10,.0f} | 标准差: ¥{expanded_df['income'].std():>10,.0f}")

# 工作年限
print(f"\n工作年限(work_years):")
print(f"  原始 - 平均: {original_df['work_years'].mean():.1f} | 标准差: {original_df['work_years'].std():.1f}")
print(f"  扩展 - 平均: {expanded_df['work_years'].mean():.1f} | 标准差: {expanded_df['work_years'].std():.1f}")

# 风险分布
print(f"\n高风险用户比例(is_risky_user):")
print(f"  原始: {original_df['is_risky_user'].mean()*100:>6.2f}%")
print(f"  扩展: {expanded_df['is_risky_user'].mean()*100:>6.2f}%")

# 职业分布
print(f"\n职业类别分布(occupation):")
print(f"  原始:")
for occ, cnt in original_df['occupation'].value_counts().head(3).items():
    print(f"    {occ}: {cnt:>5} ({cnt/len(original_df)*100:>5.1f}%)")
print(f"  扩展:")
for occ, cnt in expanded_df['occupation'].value_counts().head(3).items():
    print(f"    {occ}: {cnt:>5} ({cnt/len(expanded_df)*100:>5.1f}%)")

# 城市分布
print(f"\n城市分布(city):")
print(f"  原始 - 种类数: {original_df['city'].nunique()}")
print(f"  扩展 - 种类数: {expanded_df['city'].nunique()}")

print("\n【数据完整性检查】")
print(f"原始数据缺失值: {original_df.isnull().sum().sum()} 个")
print(f"扩展数据缺失值: {expanded_df.isnull().sum().sum()} 个")

print("\n【风险评分分布】")
print(f"  原始 - 平均: {original_df['risk_score'].mean():.3f} | 范围: [{original_df['risk_score'].min():.3f}-{original_df['risk_score'].max():.3f}]")
print(f"  扩展 - 平均: {expanded_df['risk_score'].mean():.3f} | 范围: [{expanded_df['risk_score'].min():.3f}-{expanded_df['risk_score'].max():.3f}]")

print("\n" + "="*80)
print("✅ 数据质量验证完成")
print("="*80)
