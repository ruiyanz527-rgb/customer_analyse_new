"""
KYC数据集扩展脚本
将10K数据通过多种方法扩展到50K+

包含：
1. 同义词替换扩展 (10K → 15K)
2. 回译法增强 (15K → 25K)
3. Mixup混合 (25K → 50K)
4. 特征扰动 (50K → 70K)
"""

import pandas as pd
import numpy as np
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

np.random.seed(42)
random.seed(42)

# ============================================================================
# 配置
# ============================================================================

INPUT_PATH = '/Applications/financial LLM/financial_data/kyc_rl_training_dataset.csv'
OUTPUT_PATH = '/Applications/financial LLM/financial_data/kyc_rl_training_dataset_expanded_50k.csv'

# 扩展配置
EXPANSION_CONFIG = {
    'synonym_multiplier': 1.5,      # 同义词替换倍数
    'backtrans_multiplier': 1.0,    # 回译法倍数
    'mixup_count': 15000,           # Mixup样本数
    'perturbation_multiplier': 0.4, # 特征扰动倍数
}

# ============================================================================
# 1. 同义词替换库
# ============================================================================

OCCUPATION_SYNONYMS = {
    '技术': ['IT', '开发', '编程', '系统', '算法'],
    '金融': ['银行', '证券', '保险', '投资', '理财'],
    '教育': ['培训', '学术', '研究', '科研', '教学'],
    '销售': ['商务', '营销', '客户', '渠道', '代理'],
    '制造': ['生产', '工业', '工程', '制造业', '加工'],
    '医疗': ['医药', '卫生', '健康', '诊疗', '护理'],
    '自营': ['创业', '经营', '生意', '独立', '商业'],
    '服务': ['餐饮', '旅游', '酒店', '零售', '物业'],
    '管理': ['行政', '人力', '企划', '组织', '经营'],
}

JOB_TITLE_EXPANSION = {
    '技术': {
        '工程师': ['后端工程师', '前端工程师', '全栈工程师', '运维工程师', '测试工程师'],
        'CTO': ['技术副总裁', '技术总监', '研发总监', '技术负责人'],
        '架构师': ['系统架构师', '数据架构师', '应用架构师', '基础架构师'],
        '技术总监': ['技术经理', '技术主管', '技术总管'],
        '数据科学家': ['数据分析师', '数据工程师', '算法工程师'],
    },
    '金融': {
        'CFO': ['财务总监', '财务副总', '财务负责人'],
        '投资经理': ['投资分析师', '投资总监', '基金经理'],
        '风控总监': ['风控经理', '风险管理师', '合规总监'],
        '分析师': ['高级分析师', '资深分析师', '初级分析师'],
        '交易员': ['交易主管', '高级交易员', '资深交易员'],
    },
    # ... 其他职业的职位扩展
}

CITY_EXPANSION = {
    '北京': ['北京市', '朝阳', '东城', '西城', '海淀'],
    '上海': ['上海市', '浦东', '浦西', '徐汇', '黄浦'],
    '深圳': ['深圳市', '福田', '南山', '罗湖', '宝安'],
    '杭州': ['杭州市', '余杭', '西湖', '下城', '上城'],
    '南京': ['南京市', '鼓楼', '秦淮', '建邺', '玄武'],
    '武汉': ['武汉市', '武昌', '汉口', '汉阳', '青山'],
    '西安': ['西安市', '未央', '碑林', '长安', '灞桥'],
    '成都': ['成都市', '武侯', '锦江', '青羊', '天府'],
    '广州': ['广州市', '越秀', '天河', '白云', '荔湾'],
    '苏州': ['苏州市', '工业园区', '吴中', '相城', '吴江'],
}

# ============================================================================
# 2. 数据扩展函数
# ============================================================================

class KYCDataExpander:
    """KYC数据集扩展工具"""
    
    def __init__(self, input_path: str):
        """加载原始数据"""
        print(f"📥 加载原始数据...")
        self.df = pd.read_csv(input_path)
        self.expanded_records = []
        print(f"✅ 已加载 {len(self.df):,} 条数据")
    
    def expand_with_synonyms(self) -> int:
        """
        方法1：同义词替换扩展
        
        原理：替换occupation和city中的同义词，保持其他字段不变
        效果：10K × 1.5 = 15K
        """
        print("\n" + "="*80)
        print("🔄 方法1：同义词替换扩展")
        print("="*80)
        
        added_count = 0
        
        for idx, row in self.df.iterrows():
            original_occupation = row['occupation']
            
            # 生成同义词变体
            if original_occupation in OCCUPATION_SYNONYMS:
                synonyms = OCCUPATION_SYNONYMS[original_occupation]
                
                # 为每个同义词生成一条新数据
                for synonym in synonyms[:2]:  # 每个职业生成2个变体
                    new_row = row.copy()
                    
                    # 更新职业相关的字段
                    new_row['occupation'] = original_occupation  # 保持原职业
                    # 更新KYC文本以反映同义词
                    new_row['kyc_raw_text'] = new_row['kyc_raw_text'].replace(
                        original_occupation, 
                        synonym
                    )
                    
                    self.expanded_records.append(new_row)
                    added_count += 1
        
        print(f"✅ 同义词替换新增: {added_count:,} 条")
        return added_count
    
    def expand_with_backtranslation(self) -> int:
        """
        方法2：回译法增强（模拟）
        
        原理：通过改写KYC文本来生成新样本，保持标签不变
        效果：15K × 1.0 = 15K (新增10K)
        
        注意：这里使用规则替换模拟回译效果，真实使用应调用翻译API
        """
        print("\n" + "="*80)
        print("🔁 方法2：回译法增强")
        print("="*80)
        
        added_count = 0
        
        # 文本改写模板
        text_templates = [
            lambda row: f"申请人基本信息：姓名编号{row['user_id']}'，出生于{2024 - row['age']}年，长期居住在{row['city']}。教育背景：{row['education']}。现任：{row['job_title']}@{row['occupation']}领域。年收入水平：{row['income']:,.0f}元。从业经验：{row['work_years']}年。",
            
            lambda row: f"KYC核查结果：用户{row['user_id']}' 在{row['occupation']}行业担任{row['job_title']}职务，年薪约{row['income']:,.0f}元。学历水平：{row['education']}，年龄段：{row['age']}岁左右，居住城市：{row['city']}。工作履历稳定，行业资深。",
            
            lambda row: f"尊敬的审核员：该申请人，{row['age']}岁，{row['city']}常住者，学历为{row['education']}。现职位：{row['job_title']}，所在行业：{row['occupation']}。工作年限：{row['work_years']}年。年度收入：¥{row['income']:,.0f}。信用评分：{row['sesame_score']:.0f}/850。",
        ]
        
        for idx, row in self.df.iterrows():
            # 为每条记录生成2-3个回译变体
            for template in text_templates[:2]:
                new_row = row.copy()
                new_row['kyc_raw_text'] = template(row)
                # reasoning_chain由于推理过程相同，保持不变
                self.expanded_records.append(new_row)
                added_count += 1
        
        print(f"✅ 回译法新增: {added_count:,} 条")
        return added_count
    
    def expand_with_mixup(self, n_samples: int = 10000, alpha: float = 0.3) -> int:
        """
        方法3：Mixup混合扩展
        
        原理：随机选取两条记录，在特征空间中进行线性混合
        效果：新增 10K 条样本
        
        Mixup: x_mixed = alpha * x_i + (1-alpha) * x_j
        """
        print("\n" + "="*80)
        print("🔗 方法3：Mixup混合扩展")
        print("="*80)
        
        added_count = 0
        df_len = len(self.df)
        
        for _ in range(n_samples):
            # 随机选取两条记录
            idx_i = np.random.randint(0, df_len)
            idx_j = np.random.randint(0, df_len)
            
            row_i = self.df.iloc[idx_i]
            row_j = self.df.iloc[idx_j]
            
            # 数值特征混合
            mixed_row = row_i.copy()
            
            # 混合数值特征
            numeric_cols = ['age', 'income', 'work_years', 'sesame_score', 'transaction_frequency', 'transaction_amount', 'risk_score']
            for col in numeric_cols:
                mixed_row[col] = alpha * row_i[col] + (1 - alpha) * row_j[col]
            
            # 风险标签采用多数投票或混合概率
            mixed_row['is_risky_user'] = 1 if (row_i['is_risky_user'] + row_j['is_risky_user']) >= 1 else 0
            
            # 文本字段保持原样（选择row_i的文本）
            # 或者可以生成新的混合文本
            mixed_row['kyc_raw_text'] = f"[混合样本] 部分特征参考用户{row_j['user_id']}的信息。{row_i['kyc_raw_text']}"
            
            self.expanded_records.append(mixed_row)
            added_count += 1
        
        print(f"✅ Mixup新增: {added_count:,} 条")
        return added_count
    
    def expand_with_perturbation(self, noise_ratio: float = 0.1) -> int:
        """
        方法4：特征扰动增强
        
        原理：对特征值添加小量噪声，生成相似但不同的样本
        效果：新增样本数 = 原始数据 × noise_ratio
        
        注意：只扰动数值特征，保持标签类别
        """
        print("\n" + "="*80)
        print("📊 方法4：特征扰动增强")
        print("="*80)
        
        added_count = 0
        original_size = len(self.df)
        n_perturb = int(original_size * EXPANSION_CONFIG['perturbation_multiplier'])
        
        # 扰动强度（0.05 = ±5%变化）
        perturbation_std = 0.05
        
        for idx in range(n_perturb):
            # 随机选取一条记录
            sample_idx = np.random.randint(0, original_size)
            perturbed_row = self.df.iloc[sample_idx].copy()
            
            # 添加高斯噪声到数值特征
            numeric_cols = ['age', 'income', 'work_years', 'sesame_score', 'transaction_frequency', 'transaction_amount']
            
            for col in numeric_cols:
                original_val = perturbed_row[col]
                # 噪声 = 标准值 × 扰动强度 × 高斯随机数
                noise = original_val * perturbation_std * np.random.randn()
                perturbed_row[col] = max(0, original_val + noise)
            
            # 确保年龄在合理范围
            perturbed_row['age'] = int(np.clip(perturbed_row['age'], 20, 65))
            
            # 标签保持不变（是否高风险）
            # 但重新计算风险评分
            perturbed_row['risk_score'] = min(1.0, max(0.0, 
                perturbed_row['risk_score'] + np.random.randn() * 0.05
            ))
            
            # 文本字段保持原样
            self.expanded_records.append(perturbed_row)
            added_count += 1
        
        print(f"✅ 特征扰动新增: {added_count:,} 条")
        return added_count
    
    def combine_and_save(self, output_path: str) -> int:
        """合并原始数据和扩展数据，保存为CSV"""
        print("\n" + "="*80)
        print("💾 合并与保存")
        print("="*80)
        
        # 转换expanded_records为DataFrame
        expanded_df = pd.DataFrame(self.expanded_records)
        
        # 合并原始数据和扩展数据
        combined_df = pd.concat([self.df, expanded_df], ignore_index=True)
        
        # 去重（基于user_id和主要特征）
        combined_df = combined_df.drop_duplicates(subset=['age', 'occupation', 'income', 'work_years'], keep='first')
        
        # 重新索引
        combined_df = combined_df.reset_index(drop=True)
        
        # 保存
        combined_df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"✅ 已保存到: {output_path}")
        print(f"   原始数据: {len(self.df):,} 条")
        print(f"   扩展数据: {len(self.expanded_records):,} 条")
        print(f"   最终数据: {len(combined_df):,} 条")
        
        return len(combined_df)
    
    def generate_statistics(self) -> Dict:
        """生成扩展前后的数据统计"""
        stats = {
            'original_size': len(self.df),
            'expanded_records': len(self.expanded_records),
            'total_size': len(self.df) + len(self.expanded_records),
            'original_high_risk_ratio': self.df['is_risky_user'].mean(),
            'expansion_methods': {
                'synonym_replacement': 'Data augmented via occupation synonyms',
                'backtranslation': 'Text rephrased while preserving labels',
                'mixup': 'Features linearly combined from two samples',
                'perturbation': 'Numerical features with Gaussian noise',
            }
        }
        return stats


# ============================================================================
# 3. 主程序
# ============================================================================

def main():
    """主程序：执行数据扩展"""
    
    print("\n")
    print("="*80)
    print("🚀 KYC数据集扩展工具 - 将10K扩展到50K+")
    print("="*80)
    
    # 初始化扩展器
    expander = KYCDataExpander(INPUT_PATH)
    
    # 执行扩展
    total_added = 0
    
    # 方法1：同义词替换
    added_1 = expander.expand_with_synonyms()
    total_added += added_1
    
    # 方法2：回译法
    added_2 = expander.expand_with_backtranslation()
    total_added += added_2
    
    # 方法3：Mixup混合
    added_3 = expander.expand_with_mixup(
        n_samples=EXPANSION_CONFIG['mixup_count'],
        alpha=0.3
    )
    total_added += added_3
    
    # 方法4：特征扰动
    added_4 = expander.expand_with_perturbation()
    total_added += added_4
    
    # 合并和保存
    final_size = expander.combine_and_save(OUTPUT_PATH)
    
    # 生成统计报告
    print("\n" + "="*80)
    print("📊 数据扩展统计报告")
    print("="*80)
    
    stats = expander.generate_statistics()
    
    print(f"\n【扩展摘要】")
    print(f"  原始数据规模:     {stats['original_size']:>10,} 条")
    print(f"  新增扩展数据:     {stats['expanded_records']:>10,} 条")
    print(f"  最终数据规模:     {stats['total_size']:>10,} 条")
    print(f"  扩展倍数:         {stats['total_size'] / stats['original_size']:>10.1f}x")
    print(f"\n【数据质量】")
    print(f"  原始高风险占比:   {stats['original_high_risk_ratio']:>10.2%}")
    
    print(f"\n【扩展方法】")
    for method, desc in stats['expansion_methods'].items():
        print(f"  • {method}: {desc}")
    
    print(f"\n【文件信息】")
    print(f"  输入文件:  {INPUT_PATH}")
    print(f"  输出文件:  {OUTPUT_PATH}")
    
    print("\n" + "="*80)
    print("✨ 数据扩展完毕！")
    print("="*80)
    
    # 返回统计信息
    return {
        'original': stats['original_size'],
        'expanded': stats['expanded_records'],
        'total': stats['total_size'],
        'multiplier': stats['total_size'] / stats['original_size']
    }


if __name__ == '__main__':
    result = main()
    
    # 最后的验证
    print("\n✅ 扩展成功！")
    print(f"   数据规模: {result['original']:,} → {result['total']:,}")
    print(f"   扩展倍数: {result['multiplier']:.1f}x")
    print(f"\n建议后续步骤:")
    print(f"  1. 验证数据质量: python3 -c \"import pandas as pd; df=pd.read_csv('{OUTPUT_PATH}'); print(df.describe())\"")
    print(f"  2. 重新训练模型: python3 glm4_sft_trainer.py (需要修改data_path)")
    print(f"  3. 对比性能: 检查SFT基线是否提升")
