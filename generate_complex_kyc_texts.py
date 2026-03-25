"""
生成复杂的KYC长文本特征
包含多种文本类型、复杂推理链路、多步骤决策等

适用于GSPO (长文本强化学习) 和高级RL训练场景
"""

import pandas as pd
import numpy as np
import json
import random
from typing import Dict, List, Tuple

np.random.seed(42)
random.seed(42)

# ============================================================================
# 第1部分: 复杂文本模板库
# ============================================================================

class ComplexKYCTextGenerator:
    """生成复杂的KYC长文本特征"""
    
    def __init__(self):
        # 工作经历模板
        self.work_experience_templates = [
            "申请人在{company}担任{position}一职{years}年，主要负责{responsibilities}。期间获得{achievements}，"
            "建立了{contribution}的工作方法。在{company}期间累计薪资水平达到{salary}，"
            "获得过{awards}等荣誉。离职原因为{reason}。",
            
            "{position}经验: 申请人自{start_year}年起在{industry}行业工作，在{company1}任职{years1}年，"
            "后转入{company2}，在该公司任职至今{years2}年。工作期间负责{key_responsibilities}，"
            "带领团队完成了{major_projects}等重大项目。年均绩效评分达到{performance}。",
            
            "职业发展轨迹: 从{early_position}开始职业生涯，逐步晋升至{current_position}。"
            "先后在{num_companies}家公司工作，累计工作年限{total_years}年。在{key_company}期间"
            "表现最为突出，获得了{recognition}。目前在{current_company}担任{current_role}，"
            "月薪范围为{salary_range}。",
        ]
        
        # 财务状况模板
        self.financial_status_templates = [
            "申请人年收入约{annual_income}元，月均支出{monthly_expense}元。近{months}个月的"
            "交易记录显示，月均交易{avg_transaction}笔，主要交易类型包括{transaction_types}。"
            "申请人{savings_behavior}，目前{asset_status}。房产{house_status}，车产{car_status}。",
            
            "财务概览: 申请人为{employment_type}人士，年收入在{income_range}范围内。"
            "从{bank_name}提供的交易流水来看，{months}个月内共有{total_transactions}笔交易，"
            "月均交易额度为{avg_amount}。支出结构中，{expense_breakdown}。"
            "应急储备资金约{emergency_fund}，可维持{months_reserve}个月的日常开支。",
            
            "收入分析: 申请人主要收入来源为{income_source}。{additional_income}。"
            "年收入稳定性：{stability}。过去{analysis_period}年的收入趋势{trend}。"
            "当前收入足以覆盖{expenses}的日常开支，{debt_status}。",
        ]
        
        # 社交信用评分模板
        self.credit_profile_templates = [
            "芝麻信用评分为{sesame_score}分，在{percentile}的用户中处于{rank}位置。"
            "评分构成中，身份认证占{identity_score}分，履约能力占{performance_score}分，"
            "信用历史占{history_score}分，交易行为占{behavior_score}分。"
            "过去{months}个月无逾期记录，{behavior_characteristics}。",
            
            "信用档案: {credit_history}。申请人{account_status}，目前持有{accounts}个账户。"
            "征信查询记录显示，{inquiry_frequency}。借贷记录{borrowing_history}。"
            "总体信用评级为{credit_rating}，{credit_outlook}。",
            
            "信用评估: 基于{data_sources}的多维数据评估，申请人信用评分为{score}。"
            "{positive_factors}；{negative_factors}。综合评估，申请人{overall_assessment}。",
        ]
        
        # 风险特征模板
        self.risk_feature_templates = [
            "风险识别: {risk_factor1}。申请人{behavior1}，{behavior2}。"
            "基于交易模式分析，{pattern_analysis}。{red_flag_indicators}。"
            "综合风险评分为{risk_score}，风险等级为{risk_level}。",
            
            "风险评估报告: 申请人的主要风险因素包括{risk_factors}。"
            "具体表现为：{risk_detail1}；{risk_detail2}；{risk_detail3}。"
            "建议采取{mitigation_measures}的风险控制措施。",
            
            "异常行为检测: {anomaly_detection}。{pattern_description}。"
            "与同行业、同年龄、同地区用户对标，申请人的{comparison_metrics}。"
            "因此，{risk_conclusion}。",
        ]
        
        # 贷款申请材料模板
        self.loan_application_templates = [
            "申请信息: 申请人于{apply_date}提交贷款申请，申请额度{loan_amount}。"
            "申请用途为{loan_purpose}。申请人提供了{documents_submitted}等材料。"
            "核实显示，{document_verification}。基于评估，{recommendation}。",
            
            "申请审核: 申请人提交的材料包括{materials}。"
            "经过{verification_process}的审核，{verification_results}。"
            "初步评估额度建议为{suggested_amount}，利率建议为{suggested_rate}。",
        ]
        
        # 行业分析和职业背景
        self.industry_analysis_templates = [
            "行业背景: 申请人所在{industry}行业，{industry_characteristics}。"
            "该行业平均薪资为{industry_avg_salary}，申请人的{salary_comparison}。"
            "近{period}年该行业{industry_trend}，对申请人的{impact}。",
            
            "职业分析: {occupation}这一职业，{occupation_description}。"
            "该职位{income_level}，{stability_level}。{career_prospects}。",
        ]
    
    def generate_work_experience(self, user_data: Dict) -> str:
        """生成复杂的工作经历文本"""
        # 使用第一个模板避免复杂的格式问题
        companies = ['阿里巴巴', '腾讯', '字节跳动', '百度', '美团', '滴滴', '快手', '小红书']
        positions = ['高级工程师', '产品经理', '数据分析师', '项目经理', '技术主管', '产品总监']
        responsibilities = ['系统架构设计', '产品规划', '数据分析', '团队管理', '项目交付']
        achievements = ['完成核心项目', '获得专利', '带领团队', '产品日活增长']
        contributions = ['规范化', '体系化', '流程优化', '技术创新']
        awards = ['年度最佳员工', '技术创新奖', '卓越绩效奖', '最佳项目奖']
        reasons = ['寻求更好的发展机会', '家庭原因', '创业需要', '职业转向']
        
        years1 = np.random.randint(2, 8)
        salary_range = f"¥{np.random.randint(10, 30)}K-¥{np.random.randint(35, 50)}K"
        
        text = (f"申请人在{random.choice(companies)}担任{random.choice(positions)}一职{years1}年，"
                f"主要负责{random.choice(responsibilities)}。期间获得{random.choice(achievements)}，"
                f"建立了{random.choice(contributions)}的工作方法。在公司期间累计薪资水平达到{salary_range}，"
                f"获得过{random.choice(awards)}等荣誉。离职原因为{random.choice(reasons)}。")
        
        return text
    
    def generate_financial_status(self, user_data: Dict) -> str:
        """生成复杂的财务状况文本"""
        annual_income = user_data.get('income', 20000)
        monthly_expense = int(annual_income / 12 * np.random.uniform(0.4, 0.7))
        avg_transaction = user_data.get('transaction_frequency', 30)
        transaction_types = '工资入账,日常消费,转账,投资理财'
        
        savings_behavior = random.choice(['定期存款', '理财产品投资', '基金定投', '股票投资'])
        asset_status = random.choice(['资产充足', '资产一般', '资产不足'])
        house_status = random.choice(['自有住房', '按揭房产', '租住'])
        car_status = random.choice(['有车产', '无车产', '按揭车'])
        
        months = 6
        total_transactions = int(avg_transaction * months)
        avg_amount = np.random.randint(1000, 10000)
        
        emergency_fund = annual_income * np.random.uniform(0.3, 0.8)
        months_reserve = max(1, int(emergency_fund / (monthly_expense + 1)))
        
        income_source = random.choice(['工资收入', '投资收益', '副业收入'])
        additional_income = random.choice(
            ['无额外收入', '有兼职收入，月均¥2000-5000', '有投资收益，月均¥1000-3000']
        )
        stability = random.choice(['稳定', '波动不大', '有波动'])
        trend = random.choice(['上升', '稳定', '下降'])
        debt_status = random.choice(['无负债', '有少量负债', '负债较多'])
        
        text = (f"申请人年收入约¥{int(annual_income):,.0f}元，月均支出¥{monthly_expense:,.0f}元。"
                f"近{months}个月的交易记录显示，月均交易{int(avg_transaction)}笔，"
                f"主要交易类型包括{transaction_types}。申请人{savings_behavior}，"
                f"目前{asset_status}。房产{house_status}，车产{car_status}。"
                f"年收入稳定性为{stability}，近3年收入趋势{trend}。{debt_status}。")
        
        return text
    
    def generate_credit_profile(self, user_data: Dict) -> str:
        """生成复杂的信用评分文本"""
        sesame_score = user_data.get('sesame_score', 650)
        percentile = np.random.randint(60, 95)
        
        behavior_characteristics = random.choice([
            '交易行为良好，无异常消费',
            '交易频繁，消费较为活跃',
            '消费理性，有规律的理财行为',
        ])
        
        credit_history = random.choice([
            '申请人信用记录良好，无逾期记录',
            '申请人有少量逾期记录，但均已结清',
            '申请人信用记录较为干净',
        ])
        
        accounts = np.random.randint(3, 8)
        inquiry_frequency = random.choice(['查询频繁', '查询适中', '查询较少'])
        borrowing_history = random.choice(['有借贷记录，均按时还款', '借贷记录较少', '无借贷记录'])
        credit_rating = random.choice(['优秀', '良好', '一般'])
        
        positive_factors = random.choice([
            '有良好的还款记录',
            '有稳定的收入来源',
            '有多渠道的资产',
        ])
        negative_factors = random.choice([
            '近期查询较频繁',
            '有少量逾期记录',
            '负债率略高',
        ])
        overall_assessment = random.choice([
            '信用状况良好，可信度高',
            '信用状况不错，风险较低',
            '信用状况一般，需要关注',
        ])
        
        text = (f"芝麻信用评分为{int(sesame_score)}分，在{percentile}%的用户中处于较高位置。"
                f"{credit_history}。申请人{accounts}个账户使用良好，{inquiry_frequency}。"
                f"{borrowing_history}。总体信用评级为{credit_rating}。"
                f"正面因素：{positive_factors}；负面因素：{negative_factors}。"
                f"综合评估，{overall_assessment}。")
        
        return text
    
    def generate_risk_features(self, user_data: Dict) -> str:
        """生成复杂的风险特征文本"""
        is_risky = user_data.get('is_risky_user', 0)
        risk_score = user_data.get('risk_score', 0.3)
        risk_level = '高风险' if is_risky else '低风险'
        
        if is_risky:
            risk_indicators = [
                '交易活跃度异常高',
                '频繁的大额转账',
                '消费模式不稳定',
            ]
            pattern = '交易模式显示存在异常资金流转，需要进一步核实'
            mitigation = '需要严格的额度限制和交易监控'
        else:
            risk_indicators = [
                '交易行为稳定',
                '消费模式合理',
                '资金流向清晰',
            ]
            pattern = '交易模式稳定，符合该群体的一般规律'
            mitigation = '可以进行标准授信程序'
        
        text = (f"风险识别: {risk_indicators[0]}。申请人{risk_indicators[1]}，{risk_indicators[2]}。"
                f"基于交易模式分析，{pattern}。"
                f"综合风险评分为{risk_score:.2f}，风险等级为{risk_level}。"
                f"风险控制建议：{mitigation}。")
        
        return text
    
    def generate_industry_analysis(self, user_data: Dict) -> str:
        """生成行业分析文本"""
        industry = user_data.get('occupation', '技术')
        industry_characteristics = random.choice([
            '发展前景广阔，薪资水平持续上升',
            '人才需求旺盛，竞争压力较大',
            '市场需求稳定，职位相对固定',
        ])
        
        industry_avg_salary = np.random.randint(15, 30) * 1000
        user_salary = user_data.get('income', 20000)
        
        if user_salary > industry_avg_salary:
            salary_comparison = '高于行业平均'
        elif user_salary < industry_avg_salary * 0.8:
            salary_comparison = '低于行业平均'
        else:
            salary_comparison = '接近行业平均'
        
        industry_trend = random.choice(['保持增长', '基本稳定', '有所下降'])
        occupation_description = random.choice([
            '专业性强，需要丰富的行业经验',
            '发展潜力大，薪资增长空间明显',
            '市场需求稳定，职业稳定性较好',
        ])
        
        text = (f"行业背景：申请人所在{industry}行业，{industry_characteristics}。"
                f"该行业平均薪资为¥{int(industry_avg_salary):,.0f}，"
                f"申请人的收入水平{salary_comparison}。近3年该行业{industry_trend}，"
                f"对申请人的职业发展有一定影响。该职位{occupation_description}。")
        
        return text
    
    def generate_comprehensive_kyc_text(self, user_data: Dict) -> str:
        """生成完整的综合KYC文本（长文本）"""
        
        sections = [
            f"【个人信息】\n"
            f"申请人：用户{user_data.get('user_id', 0)}\n"
            f"年龄：{user_data.get('age', 30)}岁\n"
            f"学历：{user_data.get('education', '本科')}\n"
            f"城市：{user_data.get('city', '北京')}\n\n",
            
            f"【工作背景】\n{self.generate_work_experience(user_data)}\n\n",
            
            f"【财务状况】\n{self.generate_financial_status(user_data)}\n\n",
            
            f"【信用评分】\n{self.generate_credit_profile(user_data)}\n\n",
            
            f"【风险评估】\n{self.generate_risk_features(user_data)}\n\n",
            
            f"【行业分析】\n{self.generate_industry_analysis(user_data)}\n",
        ]
        
        return ''.join(sections)
    
    def generate_multi_step_reasoning(self, user_data: Dict) -> str:
        """生成多步骤的复杂推理链路"""
        
        is_risky = user_data.get('is_risky_user', 0)
        risk_score = user_data.get('risk_score', 0.3)
        
        steps = []
        
        # Step 1: 职业和收入评估
        occupation = user_data.get('occupation', '技术')
        income = user_data.get('income', 20000)
        
        if occupation in ['自营', '服务']:
            steps.append("[职业风险评估] 申请人从事自营/服务行业，行业风险基数较高(25%)，需要重点关注收入稳定性")
        elif occupation in ['技术', '金融']:
            steps.append("[职业优势] 申请人从事技术/金融行业，收入稳定性较强，薪资增长潜力大")
        else:
            steps.append("[职业评估] 申请人职业属于中等风险行业，收入相对稳定")
        
        # Step 2: 收入和年龄匹配度
        age = user_data.get('age', 30)
        if age > 40 and income < 8000:
            steps.append("[收入警告] 申请人年龄>40岁但收入仅¥{:,.0f}，低于同龄平均水平，需要调查特殊原因".format(income))
        elif age < 25 and income > 15000:
            steps.append("[收入优势] 申请人年轻但收入较高，职业发展潜力大")
        else:
            steps.append("[收入匹配] 申请人收入与年龄基本匹配，符合预期")
        
        # Step 3: 信用记录
        sesame_score = user_data.get('sesame_score', 650)
        if sesame_score > 700:
            steps.append(f"[信用优势] 芝麻信用分{int(sesame_score)}分，高于平均水平，说明历史信用记录良好")
        elif sesame_score < 500:
            steps.append(f"[信用风险] 芝麻信用分{int(sesame_score)}分，低于平均水平，需要谨慎评估")
        else:
            steps.append(f"[信用中等] 芝麻信用分{int(sesame_score)}分，处于中等水平")
        
        # Step 4: 交易行为
        transaction_freq = user_data.get('transaction_frequency', 30)
        if transaction_freq > 50:
            steps.append("[交易活跃] 月均交易{}笔，经济活跃度高，说明资金使用频繁".format(int(transaction_freq)))
        elif transaction_freq < 10:
            steps.append("[交易稀疏] 月均交易{}笔，交易活跃度低，可能存在隐性收入或资金流向不清".format(int(transaction_freq)))
        else:
            steps.append("[交易正常] 月均交易{}笔，交易活跃度正常".format(int(transaction_freq)))
        
        # Step 5: 综合风险判断
        if is_risky == 1:
            steps.append(f"[综合评分] 综合评分{risk_score:.2f}，风险指标超过阈值，属于高风险用户")
            steps.append("[决策] 建议拒绝或严格控制额度，需要进一步的尽职调查")
        else:
            steps.append(f"[综合评分] 综合评分{risk_score:.2f}，风险指标在可控范围内，属于低风险用户")
            steps.append(f"[决策] 建议可以进行标准授信，建议授信额度为¥{int(income*3):,.0f}-¥{int(income*5):,.0f}")
        
        return '|'.join(steps)


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序：为现有数据集添加复杂长文本特征"""
    
    print("=" * 80)
    print("🚀 为KYC数据集生成复杂长文本特征")
    print("=" * 80)
    
    # 加载现有数据集
    input_path = '/Applications/financial LLM/financial_data/kyc_rl_training_dataset.csv'
    output_path = '/Applications/financial LLM/financial_data/kyc_rl_training_dataset_with_complex_text.csv'
    
    print(f"\n📥 加载原始数据集...")
    df = pd.read_csv(input_path)
    print(f"✅ 已加载 {len(df):,} 条记录")
    
    # 初始化文本生成器
    generator = ComplexKYCTextGenerator()
    
    # 生成复杂文本特征
    print(f"\n🔨 生成复杂长文本特征...")
    
    complex_texts = []
    multi_step_reasonings = []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  进度: {idx:,}/{len(df):,}")
        
        user_dict = row.to_dict()
        
        # 生成综合KYC长文本
        complex_text = generator.generate_comprehensive_kyc_text(user_dict)
        complex_texts.append(complex_text)
        
        # 生成多步骤推理链路
        multi_step_reasoning = generator.generate_multi_step_reasoning(user_dict)
        multi_step_reasonings.append(multi_step_reasoning)
    
    print(f"✅ 已生成复杂文本特征")
    
    # 添加新列到数据框
    df['kyc_complex_text'] = complex_texts
    df['reasoning_chain_detailed'] = multi_step_reasonings
    
    # 保存增强后的数据集
    print(f"\n💾 保存增强后的数据集...")
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ 已保存到: {output_path}")
    
    # 统计信息
    print(f"\n" + "=" * 80)
    print(f"📊 数据增强统计")
    print(f"=" * 80)
    print(f"原始记录数: {len(df):,}")
    print(f"新增字段: kyc_complex_text, reasoning_chain_detailed")
    print(f"总字段数: {len(df.columns)}")
    
    # 显示示例
    print(f"\n【示例：复杂KYC文本】")
    print(f"用户 {df.iloc[0]['user_id']}")
    print(df.iloc[0]['kyc_complex_text'][:500] + "...")
    
    print(f"\n【示例：多步骤推理链路】")
    print(df.iloc[0]['reasoning_chain_detailed'])
    
    # 保存样本到JSONL用于GSPO训练
    gspo_data_path = '/Applications/financial LLM/financial_data/kyc_gspo_training_data.jsonl'
    print(f"\n🔨 生成GSPO长文本训练数据...")
    
    with open(gspo_data_path, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            gspo_item = {
                'user_id': row['user_id'],
                'prompt': f"请基于以下详细的KYC材料，完成多步骤的风险评估分析：\n\n{row['kyc_complex_text']}",
                'target': row['reasoning_chain_detailed'],
                'text_length': len(row['kyc_complex_text']),
                'risk_label': row['is_risky_user'],
            }
            f.write(json.dumps(gspo_item, ensure_ascii=False) + '\n')
    
    print(f"✅ 已生成GSPO训练数据: {gspo_data_path}")
    
    print(f"\n" + "=" * 80)
    print(f"✨ 复杂长文本特征生成完毕！")
    print(f"=" * 80)


if __name__ == '__main__':
    main()
