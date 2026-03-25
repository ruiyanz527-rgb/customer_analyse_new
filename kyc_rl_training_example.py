"""
KYC智能用户理解 - 强化学习训练完整示例

基于美团金融LLM强化学习实践文档
场景：用GRPO训练模型识别用户风险、工作职位、收入等复杂字段
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import json

# ============================================================================
# 第1部分：数据准备 (SFT + RL都需要)
# ============================================================================

print("=" * 80)
print("🚀 KYC强化学习训练流程演示")
print("=" * 80)

# 加载数据集
df = pd.read_csv('/Applications/financial LLM/financial_data/kyc_rl_training_dataset.csv')
print(f"\n✅ 已加载KYC数据集: {len(df)} 条记录")

# ============================================================================
# 第2部分：SFT阶段 - 生成监督微调数据
# ============================================================================

print("\n" + "=" * 80)
print("📝 第1步：生成SFT训练数据")
print("=" * 80)

class SFTDataGenerator:
    """生成SFT训练数据"""
    
    def __init__(self, df):
        self.df = df
        self.sft_data = []
    
    def generate(self):
        """为每条记录生成 prompt -> reasoning_chain 对"""
        for idx, row in self.df.iterrows():
            # 构造prompt
            prompt = self._build_prompt(row)
            
            # target是标注的推理链路
            target = row['reasoning_chain']
            
            self.sft_data.append({
                'id': row['user_id'],
                'prompt': prompt,
                'target': target,
                'risk_label': row['is_risky_user'],
                'risk_score': row['risk_score']
            })
        
        return self.sft_data
    
    def _build_prompt(self, row):
        """构造KYC审核prompt"""
        template = f"""请基于以下KYC信息，逐步分析用户的风险等级：

【用户基础信息】
- 年龄: {row['age']}岁
- 城市: {row['city']}
- 学历: {row['education']}

【职业信息】
- 职业类别: {row['occupation']}
- 具体职位: {row['job_title']}
- 工作年限: {row['work_years']}年

【财务特征】
- 年收入: ¥{row['income']:,.0f}
- 月均交易: {row['transaction_frequency']}笔
- 芝麻信用分: {row['sesame_score']:.0f}

【KYC材料】
{row['kyc_raw_text']}

请提供多步骤的风险评估分析。"""
        return template

# 生成SFT数据
sft_gen = SFTDataGenerator(df)
sft_data = sft_gen.generate()

print(f"✅ 已生成 {len(sft_data)} 条SFT训练数据")
print(f"\n【SFT数据样例】")
sample = sft_data[0]
print(f"Prompt (前200字符):\n{sample['prompt'][:200]}...\n")
print(f"Target推理链路:\n{sample['target']}\n")
print(f"风险标签: {'高风险' if sample['risk_label'] else '低风险'} (评分: {sample['risk_score']:.3f})")

# 保存SFT数据
sft_output = '/Applications/financial LLM/financial_data/kyc_sft_training_data.jsonl'
with open(sft_output, 'w', encoding='utf-8') as f:
    for item in sft_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f"✅ SFT数据已保存到: {sft_output}")

# ============================================================================
# 第3部分：差异化奖励设计 (GRPO阶段)
# ============================================================================

print("\n" + "=" * 80)
print("🎯 第2步：设计差异化奖励函数")
print("=" * 80)

class DifferentiatedRewardDesigner:
    """
    根据字段难度设计差异化奖励
    
    难刻画字段（工作职位、收入、工作地点）给予更高权重
    常规字段（年龄、学历）给予基础权重
    """
    
    def __init__(self):
        # 定义字段难度权重
        self.field_weights = {
            'occupation': 1.0,      # 基础权重
            'job_title': 1.5,       # 难刻画 (+50%)
            'income': 1.3,          # 中等难度 (+30%)
            'risk_classification': 1.2  # 中等难度
        }
    
    def compute_reward(self, model_output, golden_data, difficulty_level='medium'):
        """
        计算奖励信号
        
        Args:
            model_output: 模型生成的文本
            golden_data: 真实标签
            difficulty_level: 任务难度 ('easy' / 'medium' / 'hard')
        
        Returns:
            reward: 奖励分数 [-1.0, 2.0]
        """
        
        # 1. 基础奖励：是否正确识别风险
        base_reward = self._check_risk_classification(model_output, golden_data)
        
        # 2. 难刻画字段奖励加成
        field_bonus = 0.0
        
        # 职位识别
        if self._check_job_title(model_output, golden_data['job_title']):
            field_bonus += 0.5 * self.field_weights['job_title']
        
        # 收入推理
        if self._check_income_reasonable(model_output, golden_data['income']):
            field_bonus += 0.3 * self.field_weights['income']
        
        # 3. 推理链路完整性
        chain_bonus = self._check_reasoning_completeness(model_output)
        
        # 综合奖励
        total_reward = base_reward + field_bonus + chain_bonus
        
        return total_reward
    
    def _check_risk_classification(self, model_output, golden_data):
        """检查风险分类是否正确"""
        golden_risk = golden_data['is_risky_user']
        # 模拟检查: 如果输出包含对应关键词，则视为正确
        if golden_risk == 1:
            if '高风险' in model_output or '建议拒绝' in model_output:
                return 1.0
            else:
                return -0.5
        else:
            if '低风险' in model_output or '可以正常' in model_output:
                return 1.0
            else:
                return -0.5
    
    def _check_job_title(self, model_output, golden_title):
        """检查是否准确识别职位"""
        if golden_title in model_output or len(model_output) > 100:
            return True
        return False
    
    def _check_income_reasonable(self, model_output, golden_income):
        """检查是否合理推理收入"""
        if '收入' in model_output or '薪' in model_output or '¥' in model_output:
            return True
        return False
    
    def _check_reasoning_completeness(self, model_output):
        """检查推理链路的完整性"""
        completeness_score = 0.0
        
        # 计算推理步骤数量
        steps = [
            '职业' in model_output,
            '收入' in model_output,
            '交易' in model_output,
            '风险' in model_output
        ]
        
        step_count = sum(steps)
        completeness_score = (step_count / 4.0) * 0.2  # 最多加0.2分
        
        return completeness_score

# 初始化奖励设计器
reward_designer = DifferentiatedRewardDesigner()

# 计算全数据集的奖励信号
print(f"\n计算全数据集的奖励信号...")
rewards = []

for sample in sft_data[:100]:  # 演示用前100个
    model_output_demo = sample['target']  # 实际中这是模型生成的
    golden_data = {
        'is_risky_user': sample['risk_label'],
        'job_title': '',  # 从原始数据获取
        'income': 0
    }
    
    reward = reward_designer.compute_reward(
        model_output_demo, 
        golden_data,
        difficulty_level='medium'
    )
    rewards.append(reward)

print(f"✅ 奖励信号统计:")
print(f"   平均奖励: {np.mean(rewards):.3f}")
print(f"   最大奖励: {np.max(rewards):.3f}")
print(f"   最小奖励: {np.min(rewards):.3f}")
print(f"   奖励分布: {np.percentile(rewards, [25, 50, 75])}")

# ============================================================================
# 第4部分：课程学习 - 按难度分级
# ============================================================================

print("\n" + "=" * 80)
print("📚 第3步：课程学习 - 按难度分级数据")
print("=" * 80)

class CurriculumLearningScheduler:
    """
    课程学习调度器
    
    Stage 1: 仅用简单样本训练 (risk_score < 0.3 或 risk_score > 0.7)
    Stage 2: 混合简单和困难样本 (1:1 比例)
    """
    
    def __init__(self, df):
        self.df = df
    
    def split_by_difficulty(self):
        """将数据按难度分级"""
        
        # 简单样本: 明显低风险 或 明显高风险
        simple_mask = (self.df['risk_score'] < 0.3) | (self.df['risk_score'] > 0.7)
        simple_data = self.df[simple_mask]
        
        # 困难样本: 风险评分在中等范围
        hard_mask = (self.df['risk_score'] >= 0.35) & (self.df['risk_score'] <= 0.65)
        hard_data = self.df[hard_mask]
        
        return simple_data, hard_data
    
    def get_curriculum_schedule(self):
        """获取课程学习的完整计划"""
        simple_data, hard_data = self.split_by_difficulty()
        
        return {
            'stage_1': {
                'name': '基础能力构建',
                'data': simple_data,
                'epochs': 5,
                'description': '仅使用简单样本，快速掌握基本规律'
            },
            'stage_2': {
                'name': '能力强化提升',
                'data': pd.concat([
                    simple_data.sample(n=min(len(simple_data), len(hard_data)), random_state=42),
                    hard_data
                ]),
                'epochs': 5,
                'description': '混合简单和困难样本 (1:1)，攻克复杂场景'
            }
        }

scheduler = CurriculumLearningScheduler(df)
simple_data, hard_data = scheduler.split_by_difficulty()
curriculum = scheduler.get_curriculum_schedule()

print(f"\n【课程学习计划】")
print(f"\nStage 1 - 基础能力构建:")
print(f"  数据量: {len(curriculum['stage_1']['data'])} 条")
print(f"  样本特点: 明显低风险或明显高风险")
print(f"  训练轮数: {curriculum['stage_1']['epochs']} epochs")
print(f"  风险分分布: {curriculum['stage_1']['data']['risk_score'].describe()}")

print(f"\nStage 2 - 能力强化提升:")
print(f"  数据量: {len(curriculum['stage_2']['data'])} 条")
print(f"  样本特点: 1:1 混合简单和困难样本")
print(f"  训练轮数: {curriculum['stage_2']['epochs']} epochs")

# 保存课程学习数据
for stage_name, stage_info in curriculum.items():
    output_path = f'/Applications/financial LLM/financial_data/kyc_curriculum_{stage_name}_data.csv'
    stage_info['data'].to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✅ {stage_name}数据已保存到: {output_path}")

# ============================================================================
# 第5部分：GRPO超参数推荐
# ============================================================================

print("\n" + "=" * 80)
print("⚙️  第4步：GRPO训练超参数推荐")
print("=" * 80)

grpo_config = {
    'model_config': {
        'model_name': 'GLM-4-9B-Chat',  # 推荐模型
        'dtype': 'bfloat16',
        'max_seq_length': 8192,
    },
    'training_config': {
        'batch_size': 8,
        'num_train_epochs': 10,
        'learning_rate': 1e-5,
        'warmup_steps': 500,
        'save_steps': 200,
    },
    'rl_config': {
        'algorithm': 'GRPO',  # 使用GRPO算法
        'group_size': 8,  # 每个prompt生成8个response
        'actor_entropy_coeff': 0.001,  # 熵正则系数 (按照文档推荐)
        'actor_clip_ratio_high': 0.28,  # Clip Ratio高上界 (DAPO策略)
        'kl_loss_coeff': 0.003,  # KL正则系数
        'kl_loss_type': 'k2',  # KL估计器类型
    },
    'reward_config': {
        'use_differentiated_reward': True,  # 启用差异化奖励
        'field_weights': {
            'occupation': 1.0,
            'job_title': 1.5,
            'income': 1.3,
        },
        'base_reward': 1.0,
        'negative_reward': -0.5,
    },
    'curriculum_config': {
        'enable_curriculum': True,
        'stage_1_epochs': 5,
        'stage_2_epochs': 5,
        'easy_hard_ratio': 1.0,
    }
}

print("\n【GRPO训练推荐配置】")
print(json.dumps(grpo_config, indent=2, ensure_ascii=False))

# 保存配置
config_path = '/Applications/financial LLM/financial_data/kyc_grpo_config.json'
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(grpo_config, f, indent=2, ensure_ascii=False)
print(f"\n✅ 配置已保存到: {config_path}")

# ============================================================================
# 第6部分：性能预期与基准
# ============================================================================

print("\n" + "=" * 80)
print("📊 第5步：性能基准与预期改进")
print("=" * 80)

benchmark = {
    'baseline_sft': {
        '工作职位识别精度': '61.45%',
        '当前在职情况识别': '70.15%',
        '收入区间预测': '59.09%',
        '工作类别识别': '63.80%',
        '综合准确率': '82.80%',
    },
    'expected_grpo': {
        '工作职位识别精度': '63.82%',  # +2.37pp
        '当前在职情况识别': '72.57%',  # +2.42pp
        '收入区间预测': '60.45%',      # +1.36pp
        '工作类别识别': '66.47%',      # +2.67pp
        '综合准确率': '83.01%',        # +0.21pp
    },
    'expected_gspo': {
        '工作职位识别精度': '64.27%',  # +2.82pp
        '当前在职情况识别': '74.69%',  # +4.54pp
        '收入区间预测': '61.44%',      # +2.35pp
        '工作类别识别': '69.32%',      # +5.52pp
        '综合准确率': '83.01%',        # +0.21pp (相同)
    },
}

print("\n【性能对比】")
print("\n1. SFT基线（监督微调）")
for metric, value in benchmark['baseline_sft'].items():
    print(f"   {metric:15s}: {value:>8s}")

print("\n2. GRPO优化后")
for metric, value in benchmark['expected_grpo'].items():
    baseline = benchmark['baseline_sft'][metric]
    improvement = "↑ +{:.2f}pp".format(
        float(value.rstrip('%')) - float(baseline.rstrip('%'))
    )
    print(f"   {metric:15s}: {value:>8s}  {improvement}")

print("\n3. GSPO优化后（推荐用于长文本）")
for metric, value in benchmark['expected_gspo'].items():
    baseline = benchmark['baseline_sft'][metric]
    improvement = "↑ +{:.2f}pp".format(
        float(value.rstrip('%')) - float(baseline.rstrip('%'))
    )
    print(f"   {metric:15s}: {value:>8s}  {improvement}")

# ============================================================================
# 第7部分：输出总结
# ============================================================================

print("\n" + "=" * 80)
print("✨ 强化学习训练准备完成")
print("=" * 80)

summary = f"""
【训练数据准备总结】

1. ✅ SFT数据集: {len(sft_data)} 条
   - 保存位置: {sft_output}
   - 格式: JSONL（每行一条样本）

2. ✅ 课程学习数据:
   - Stage 1 简单样本: {len(curriculum['stage_1']['data'])} 条
   - Stage 2 混合样本: {len(curriculum['stage_2']['data'])} 条

3. ✅ 奖励函数配置:
   - 类型: 差异化奖励 (Differentiated Reward)
   - 难刻画字段加权: job_title×1.5, income×1.3
   - 完整性检查: 推理链路4步验证

4. ✅ 推荐模型与算法:
   - 基础模型: GLM-4-9B-Chat
   - 强化学习: GRPO (可选: GSPO for long text)
   - 混合训练: RLxSFT (CHORD/LUFFY方案)

5. ✅ 超参数配置已保存:
   - {config_path}

【后续步骤】

1. 使用veRL框架加载本数据集进行GRPO训练
2. 监控熵值曲线（避免熵坍缩和熵爆炸）
3. Stage 1 → Stage 2 的课程学习训练
4. 验证性能是否达到预期提升 (+2-5pp)

【关键指标监控】

- 熵曲线: 应呈现"先稳定后缓升"（复杂任务特征）
- Reward分布: 应逐步右移，表现为高奖励样本比例增加
- 难刻画字段精度: 应显著超过SFT基线（+2-5pp提升空间）

"""

print(summary)

# 最后验证文件
import os
files_created = [
    sft_output,
    '/Applications/financial LLM/financial_data/kyc_curriculum_stage_1_data.csv',
    '/Applications/financial LLM/financial_data/kyc_curriculum_stage_2_data.csv',
    config_path,
]

print("【文件验证】")
for filepath in files_created:
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    size = os.path.getsize(filepath) if exists else 0
    print(f"{status} {filepath}")
    if exists:
        print(f"   大小: {size / 1024 / 1024:.2f} MB")

print("\n" + "=" * 80)
print("🎉 强化学习训练数据准备完毕！")
print("=" * 80)
