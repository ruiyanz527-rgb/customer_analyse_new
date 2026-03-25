"""
Qwen2-7B-Instruct GRPO强化学习训练脚本
实现Group Supervised Policy Optimization用于KYC风险评估

GRPO特点:
  - 基于分组的监督学习
  - 细粒度奖励信号
  - 对齐推理链路
  - 提升推理准确性
"""

import os
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from torch.optim import AdamW
# import wandb  # 已禁用，使用本地日志
from tqdm import tqdm


# ============================================================================
# 配置
# ============================================================================

@dataclass
class QwenGRPOConfig:
    """Qwen2 GRPO训练配置（含训练稳定性参数）"""
    
    # 模型
    model_name: str = "Qwen/Qwen2-7B-Instruct"
    sft_model_path: str = "./qwen2_kyc_model"  # SFT微调后的模型路径
    
    # 数据
    data_path: str = "/Applications/financial LLM/financial_data/kyc_gspo_training_data_1_10.jsonl"
    max_seq_length: int = 512
    
    # GRPO特定参数
    group_size: int = 4  # 每组的样本数
    top_k_ranking: int = 3  # 评分的前K个
    reward_scale: float = 1.0  # 奖励缩放因子
    
    # 训练
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1  # 多卡模式：降低单卡batch size
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2  # 减少梯度累积步数（多卡后显存压力小）
    learning_rate: float = 5e-6  # GRPO通常用更小的学习率
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # -------------------------------------------------------------------------
    # 训练稳定性参数（规避熵坍缩/爆炸，在 GRPO 阶段生效）
    # -------------------------------------------------------------------------
    
    # --- Clip Ratio（PPO-style 策略梯度裁剪，非对称区间）---
    # 限制单步策略更新幅度，防止策略崩塌（policy collapse）
    #
    # 对应 veRL 框架参数：
    #   actor_rollout_ref.actor.clip_ratio      → clip_ratio      (下界偏移量 ε)
    #   actor_rollout_ref.actor.clip_ratio_high → clip_ratio_high (上界绝对值)
    #
    # 非对称裁剪区间：[1 - clip_ratio, clip_ratio_high] = [0.85, 0.28]
    #   下界 1 - 0.15 = 0.85：防止策略概率比过小（避免遗忘）
    #   上界 0.28：比对称上界 1+0.15=1.15 更保守，DAPO 策略设计
    #             正激励方向额外收紧，降低奖励 hack 风险
    clip_ratio: float = 0.15

    # clip_ratio_high：裁剪上界（对应 veRL actor_rollout_ref.actor.clip_ratio_high）
    # 推荐值 0.28，实现非对称区间 [1-clip_ratio, clip_ratio_high] = [0.85, 0.28]
    clip_ratio_high: float = 0.28
    
    # --- KL 散度惩罚 ---
    # 防止策略偏离 SFT 参考模型过远，保留已学到的格式和语言能力
    # 公式：loss += kl_coef * KL(π_θ || π_ref)
    #
    # 推荐值梯度（结合 k2 估计器）：
    #   0.001 — 宽松约束，探索空间大，适合奖励信号强且模型已较好对齐的场景
    #   0.003 — 默认推荐，平衡探索与保守，veRL/GRPO 社区常用值         ← 当前设置
    #   0.005 — 中等约束，适合任务难度高、希望保留更多 SFT 格式能力的场景
    #
    # 调参建议：
    #   - 若训练中 KL 散度持续 >0.1，考虑适当增大（如 0.003→0.005）
    #   - 若奖励提升缓慢、策略几乎不更新，考虑适当减小（如 0.003→0.001）
    kl_coef: float = 0.003
    
    # KL 估计器选择（影响计算精度与方差的权衡）：
    #   "k1" - 一阶估计: KL ≈ log(π_ref/π_θ) = ref_logprob - policy_logprob
    #           低方差，有轻微正偏，计算极快
    #   "k2" - 二阶估计（veRL/GRPO 推荐）：
    #           KL ≈ (r - 1)²/2，r = π_θ/π_ref
    #           对称二阶近似，无偏且方差比 k3 更低，数值稳定
    #           适合 KYC 长文本场景（r 偏离 1 不大时精度最优）
    #   "k3" - 三阶无偏估计: KL ≈ (r-1) - log(r)，r=π_θ/π_ref，恒≥0，方差稍大
    #   推荐：使用 "k2"（veRL 框架默认，GRPO 推荐），兼顾无偏性与低方差
    kl_estimator: str = "k2"
    
    # --- 熵正则（Entropy Regularization） ---
    # 鼓励策略保持输出多样性，防止熵坍缩（模型退化为固定模板输出）
    # 公式：loss -= entropy_coef * H(π_θ)  （最大化熵 = 最小化负熵）
    # 正值 → 奖励探索；0.0 → 关闭；负值 → 惩罚熵（不推荐）
    entropy_coef: float = 0.01
    
    # 熵告警阈值（用于检测熵坍缩/爆炸）
    # 当 mean_entropy < entropy_low_threshold  → 熵坍缩警告（模型过度确定）
    # 当 mean_entropy > entropy_high_threshold → 熵爆炸警告（模型输出混乱）
    entropy_low_threshold: float = 0.5    # 低于此值告警（坍缩）
    entropy_high_threshold: float = 8.0   # 高于此值告警（爆炸）
    
    # --- 梯度裁剪 ---
    # 覆盖 GRPO 特定的梯度裁剪阈值（比 SFT 的 1.0 更保守）
    max_grad_norm: float = 0.5
    
    # 输出
    output_dir: str = "./qwen2_kyc_grpo_model"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100


# ============================================================================
# GRPO奖励函数
# ============================================================================

class GRPORewardCalculator:
    """GRPO奖励计算器 - 差异化奖励机制
    
    设计理念：
      - 难刻画字段（工作职位/行业语义、决策）：赋予显著更高的奖励权重，
        强行拉升模型在复杂语义识别上的注意力，使梯度更新更具针对性。
      - 常规字段（学历、年龄收入匹配、信用评分、交易笔数等）：设置基础奖励系数，
        确保基本盘的稳定性，防止过度优化导致已有能力退化。
    """
    
    def __init__(self, config: QwenGRPOConfig):
        self.config = config
        
        # ----------------------------------------------------------------
        # 字段差异化奖励权重
        # 键：字段名关键词（用于匹配 [职业*]、[决策] 等步骤前缀）
        # 值：奖励系数
        # ----------------------------------------------------------------
        self.field_reward_weights: Dict[str, float] = {
            # --- 难刻画字段：语义复杂，需要行业知识→风险类型的深层映射 ---
            "职业": 2.5,   # 行业/职位语义→风险等级映射，是最难对齐的字段
            "决策": 2.0,   # 推理链末端综合判断，方向性错误代价最高
            # --- 中等难度字段：存在边界模糊，但有明确数值锚点 ---
            "综合评分": 1.2,  # 数值阈值判断（0.4分界），存在临界案例
            # --- 常规字段：语义直接，数值可直接读取，保持基础稳定性 ---
            "收入": 1.0,    # 收入与年龄/行业匹配，规则相对明确
            "信用": 1.0,    # 芝麻信用分数值直读 + 等级映射
            "交易": 1.0,    # 月均交易笔数判断，规则直接
        }
    
    def calculate_reasoning_chain_score(self, prediction: str, target: str) -> float:
        """
        计算推理链路的匹配分数（差异化奖励版本）
        
        评分组成：
          - 字段级差异化奖励 (60%)：按字段类型施加不同权重，重点奖励难刻画字段的精准对齐
          - 逻辑顺序一致性  (20%)：推理步骤的顺序合理性
          - 决策准确性      (20%)：最终决策方向的正确性（单独保留以双重强化难字段）
        """
        # 字段级差异化奖励 (60%) — 核心奖励信号
        field_reward = self.calculate_field_level_reward(prediction, target)
        
        # 逻辑顺序一致性 (20%)
        pred_steps = self._extract_steps(prediction)
        logic_score = self._calculate_logic_consistency(pred_steps)
        
        # 决策准确性 (20%) — 对难刻画的决策字段进行二次强化
        decision_score = self._calculate_decision_accuracy(prediction, target)
        
        total_score = (
            field_reward  * 0.60 +
            logic_score   * 0.20 +
            decision_score * 0.20
        )
        
        return min(total_score, 1.0)
    
    # ================================================================
    # 差异化字段级奖励（核心新增逻辑）
    # ================================================================
    
    def calculate_field_level_reward(self, prediction: str, target: str) -> float:
        """
        字段级差异化奖励计算：
          1. 将预测结果和标准答案分别拆解为若干 [字段] 步骤
          2. 对每个字段按其奖励权重进行加权评分
          3. 加权平均得到最终奖励分数
        
        当模型在难刻画字段（职业、决策）上精准对齐时，
        其权重倍数使梯度更新向这些字段倾斜，形成显著的激励偏差。
        """
        pred_fields = self._parse_fields(prediction)
        target_fields = self._parse_fields(target)
        
        if not target_fields:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for field_name, target_content in target_fields.items():
            weight = self._get_field_weight(field_name)
            pred_content = pred_fields.get(field_name, "")
            score = self._match_field_content(field_name, pred_content, target_content)
            total_weighted_score += score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def calculate_field_level_reward_detailed(self, prediction: str, target: str) -> Dict[str, float]:
        """
        返回各字段的详细奖励分数（用于日志监控）
        
        返回格式: {"职业": 0.8, "收入": 1.0, "信用": 0.9, ...}
        """
        pred_fields = self._parse_fields(prediction)
        target_fields = self._parse_fields(target)
        
        detail: Dict[str, float] = {}
        for field_name, target_content in target_fields.items():
            pred_content = pred_fields.get(field_name, "")
            detail[field_name] = self._match_field_content(field_name, pred_content, target_content)
        
        return detail
    
    def _parse_fields(self, text: str) -> Dict[str, str]:
        """
        解析推理链文本，提取各字段名称与内容。
        
        输入示例：
          "[职业优势] 从事技术行业，收入稳定 | [收入匹配] 收入与年龄匹配 | [决策] 建议标准授信"
        
        输出：
          {"职业": "从事技术行业，收入稳定", "收入": "收入与年龄匹配", "决策": "建议标准授信"}
        """
        fields: Dict[str, str] = {}
        parts = text.split('|')
        for part in parts:
            part = part.strip()
            if '[' in part and ']' in part:
                # 提取方括号内的字段标签，例如 "职业优势"、"收入匹配"
                bracket_start = part.index('[')
                bracket_end = part.index(']')
                field_label = part[bracket_start + 1:bracket_end].strip()
                # 字段内容为方括号之后的文字
                field_content = part[bracket_end + 1:].strip()
                # 归一化字段名（取核心关键词）
                normalized = self._normalize_field_name(field_label)
                if normalized:
                    fields[normalized] = field_content
        return fields
    
    def _normalize_field_name(self, label: str) -> str:
        """
        将字段标签归一化为统一的核心关键词，便于权重查找。
        
        例如："职业优势" / "职业风险评估" / "职业评估" → "职业"
              "收入匹配" / "收入优势"                → "收入"
              "信用中等" / "信用优势" / "信用风险"   → "信用"
        """
        # 按优先级顺序匹配，避免歧义
        priority_order = ["综合评分", "决策", "职业", "收入", "信用", "交易"]
        for key in priority_order:
            if key in label:
                return key
        # 未能识别的字段标签，返回原始标签（权重默认1.0）
        return label
    
    def _get_field_weight(self, field_name: str) -> float:
        """
        获取字段的奖励权重。
        优先在 field_reward_weights 中精确查找，未命中则返回基础系数 1.0。
        """
        return self.field_reward_weights.get(field_name, 1.0)
    
    def _match_field_content(self, field_name: str, pred_content: str, target_content: str) -> float:
        """
        字段内容匹配评分（0.0 ~ 1.0）。
        
        不同字段采用不同的匹配策略：
        
        [职业] 难刻画字段：
          - 检验行业风险方向一致性（高风险/中风险/低风险行业三分类）
          - 完全匹配得 1.0；方向一致但措辞不同得 0.7；方向相反得 0.0
          - 若预测为空（模型未生成该步骤）得 0.0
        
        [决策] 难刻画字段：
          - 严格区分两大方向：拒绝/严格控制 vs 标准授信
          - 方向完全正确得 1.0；部分命中（提到但不完整）得 0.4；
            方向相反得 -0.2（负奖励，惩罚模型输出错误决策方向）
        
        [常规字段]（收入/信用/交易/综合评分）：
          - 使用关键词 Jaccard 相似度评分（安全稳定，不会产生负奖励）
        """
        if not pred_content and not target_content:
            return 1.0
        if not pred_content:  # 模型完全未生成该字段
            return 0.0
        
        if field_name == "职业":
            return self._match_occupation_field(pred_content, target_content)
        elif field_name == "决策":
            return self._match_decision_field(pred_content, target_content)
        else:
            # 常规字段：关键词重叠度（Jaccard相似度）
            return self._keyword_jaccard_score(pred_content, target_content)
    
    def _match_occupation_field(self, pred: str, target: str) -> float:
        """
        职业字段专用匹配：行业风险方向一致性检验。
        
        行业风险分类：
          - 高风险行业关键词：自营、服务业、销售、餐饮、建筑
          - 低风险行业关键词：技术、金融、医疗、教育、公务员、管理
          - 中等风险行业关键词：制造、物流、零售
        """
        high_risk_keywords = {"自营", "服务", "销售", "餐饮", "建筑", "风险基数", "重点关注"}
        low_risk_keywords  = {"技术", "金融", "医疗", "教育", "公务", "管理", "稳定性较强", "增长潜力", "优势"}
        mid_risk_keywords  = {"制造", "物流", "零售", "中等风险", "相对稳定"}
        
        def classify_risk(text: str) -> str:
            """将文本分类为 high / mid / low 风险"""
            text_chars = set(text)
            high_score = sum(1 for kw in high_risk_keywords if kw in text)
            low_score  = sum(1 for kw in low_risk_keywords  if kw in text)
            mid_score  = sum(1 for kw in mid_risk_keywords  if kw in text)
            if high_score > low_score and high_score > mid_score:
                return "high"
            elif low_score > high_score and low_score > mid_score:
                return "low"
            elif mid_score > 0:
                return "mid"
            return "unknown"
        
        pred_risk  = classify_risk(pred)
        target_risk = classify_risk(target)
        
        if pred_risk == "unknown" or target_risk == "unknown":
            # 无法明确分类时，回退到关键词重叠度
            return self._keyword_jaccard_score(pred, target)
        
        if pred_risk == target_risk:
            return 1.0   # 行业风险方向完全一致
        
        # 方向部分接近（如 mid 误判为 low）给部分分
        adjacent_pairs = {("mid", "low"), ("low", "mid"), ("mid", "high"), ("high", "mid")}
        if (pred_risk, target_risk) in adjacent_pairs:
            return 0.4
        
        return 0.0  # 方向完全相反（high vs low），不给分
    
    def _match_decision_field(self, pred: str, target: str) -> float:
        """
        决策字段专用匹配：严格方向性评分，包含负奖励惩罚。
        
        两大决策方向：
          A. 拒绝/严格控制："拒绝"、"严格控制"、"拒绝或严格控制"
          B. 标准授信："标准授信"、"标准授信程序"、"建议授信"
        
        评分规则：
          - 方向正确且措辞完整：1.0
          - 方向正确但措辞不完整（只有一个关键词）：0.6
          - 方向相反（预测B但目标A，或反之）：-0.2（负奖励惩罚）
          - 无法判断方向：回退到关键词 Jaccard 相似度
        """
        reject_keywords   = {"拒绝", "严格控制", "尽职调查"}
        approve_keywords  = {"标准授信", "授信额度", "标准授信程序"}
        
        def get_direction(text: str) -> str:
            has_reject  = any(kw in text for kw in reject_keywords)
            has_approve = any(kw in text for kw in approve_keywords)
            if has_reject and not has_approve:
                return "reject"
            elif has_approve and not has_reject:
                return "approve"
            elif has_reject and has_approve:
                # 含混输出，模型不确定，给低分
                return "mixed"
            return "unknown"
        
        pred_dir   = get_direction(pred)
        target_dir = get_direction(target)
        
        if target_dir == "unknown":
            return self._keyword_jaccard_score(pred, target)
        
        if pred_dir == target_dir:
            # 方向一致：根据关键词完整度给分
            pred_keywords   = reject_keywords if pred_dir == "reject" else approve_keywords
            kw_hits = sum(1 for kw in pred_keywords if kw in pred)
            if kw_hits >= 2:
                return 1.0   # 完整命中
            else:
                return 0.6   # 方向正确但措辞不完整
        elif pred_dir == "mixed":
            return 0.2  # 含混输出，说明模型犹豫
        elif pred_dir == "unknown":
            return 0.0  # 未生成决策
        else:
            # 方向相反：给予负奖励惩罚，强迫模型关注决策方向
            return -0.2
    
    def _keyword_jaccard_score(self, pred: str, target: str) -> float:
        """
        基于字符级 n-gram 的 Jaccard 相似度，用于常规字段的模糊匹配。
        使用二元字符（bigram）提高中文匹配精度。
        """
        def get_bigrams(text: str) -> set:
            # 去除空白后取所有相邻二字对
            text = text.replace(" ", "").replace("\n", "")
            return {text[i:i+2] for i in range(len(text) - 1)} if len(text) >= 2 else set(text)
        
        pred_bigrams   = get_bigrams(pred)
        target_bigrams = get_bigrams(target)
        
        if not target_bigrams:
            return 1.0 if not pred_bigrams else 0.0
        
        intersection = len(pred_bigrams & target_bigrams)
        union        = len(pred_bigrams | target_bigrams)
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_steps(self, text: str) -> List[str]:
        """提取推理步骤"""
        steps = []
        parts = text.split('|')
        for part in parts:
            if '[' in part and ']' in part:
                steps.append(part.strip())
        return steps
    
    def _calculate_step_match(self, pred_steps: List[str], target_steps: List[str]) -> float:
        """计算步骤匹配度"""
        if len(target_steps) == 0:
            return 0.0
        
        # 步骤数匹配
        step_count_match = min(len(pred_steps), len(target_steps)) / len(target_steps)
        
        # 步骤内容相似度 (模糊匹配)
        content_match = 0.0
        for target_step in target_steps:
            for pred_step in pred_steps:
                # 检查是否包含相同的关键词
                if self._steps_similar(pred_step, target_step):
                    content_match += 1.0
                    break
        
        content_match /= len(target_steps)
        
        return (step_count_match + content_match) / 2.0
    
    def _steps_similar(self, step1: str, step2: str) -> bool:
        """检查两个步骤是否相似"""
        # 提取步骤名称 (在 [] 内)
        def get_step_name(s):
            if '[' in s and ']' in s:
                return s[s.index('[')+1:s.index(']')]
            return ""
        
        name1 = get_step_name(step1)
        name2 = get_step_name(step2)
        
        # 检查是否包含相同的关键词
        keywords1 = set(name1.split())
        keywords2 = set(name2.split())
        
        if len(keywords1 & keywords2) > 0:
            return True
        
        return False
    
    def _calculate_logic_consistency(self, steps: List[str]) -> float:
        """计算逻辑一致性"""
        if len(steps) < 2:
            return 0.5
        
        # 检查步骤顺序的合理性
        # 预期顺序: 职业 -> 收入 -> 信用 -> 交易 -> 综合 -> 决策
        expected_order = ['职业', '收入', '信用', '交易', '综合', '决策']
        
        consistency = 0.0
        for i, step in enumerate(steps):
            for j, expected in enumerate(expected_order):
                if expected in step and i <= j:
                    consistency += 1.0
        
        return min(consistency / len(steps), 1.0)
    
    def _calculate_decision_accuracy(self, prediction: str, target: str) -> float:
        """计算决策准确性"""
        # 检查最终决策是否包含相同的关键词
        
        # 提取决策关键词
        pred_keywords = set()
        target_keywords = set()
        
        # 检查风险等级
        for level in ['低风险', '高风险', 'low risk', 'high risk']:
            if level in prediction:
                pred_keywords.add('risk_level')
            if level in target:
                target_keywords.add('risk_level')
        
        # 检查建议
        for suggestion in ['标准授信', '严格控制', '拒绝', 'standard', 'strict', 'reject']:
            if suggestion in prediction:
                pred_keywords.add('suggestion')
            if suggestion in target:
                target_keywords.add('suggestion')
        
        if len(target_keywords) == 0:
            return 0.5
        
        match = len(pred_keywords & target_keywords) / len(target_keywords)
        
        return match
    
    def calculate_group_reward(self, group_predictions: List[str], group_targets: List[str]) -> List[float]:
        """
        计算组内的奖励
        
        实现ranking-based奖励: 组内表现更好的样本获得更高奖励
        """
        
        if len(group_predictions) != len(group_targets):
            raise ValueError("predictions和targets数量不匹配")
        
        # 计算每个样本的基础分数
        scores = [
            self.calculate_reasoning_chain_score(pred, target)
            for pred, target in zip(group_predictions, group_targets)
        ]
        
        # 进行排序，给予排名奖励
        sorted_indices = np.argsort(scores)[::-1]  # 从高到低排序
        
        # 生成ranking-based奖励
        rewards = np.zeros(len(scores))
        for rank, idx in enumerate(sorted_indices[:self.config.top_k_ranking]):
            rewards[idx] = (self.config.top_k_ranking - rank) / self.config.top_k_ranking
        
        # 缩放奖励
        rewards = rewards * self.config.reward_scale
        
        return rewards.tolist()


# ============================================================================
# 数据加载
# ============================================================================

class GRPODataProcessor:
    """GRPO数据处理器"""
    
    def __init__(self, config: QwenGRPOConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
    
    def load_data(self) -> List[Dict]:
        """加载训练数据"""
        print(f"📥 加载数据: {self.config.data_path}")
        
        data = []
        with open(self.config.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        print(f"✅ 已加载 {len(data):,} 条记录")
        return data
    
    def create_groups(self, data: List[Dict]) -> List[List[Dict]]:
        """创建数据组"""
        print(f"📊 创建组 (组大小: {self.config.group_size})...")
        
        # 按风险标签分组，确保每组内有混合标签
        groups = []
        low_risk = [d for d in data if d['risk_label'] == 0]
        high_risk = [d for d in data if d['risk_label'] == 1]
        
        # 交替取样
        group_idx = 0
        while group_idx * self.config.group_size < len(data):
            group = []
            for i in range(self.config.group_size):
                if len(low_risk) > 0:
                    group.append(low_risk.pop(0))
                elif len(high_risk) > 0:
                    group.append(high_risk.pop(0))
            
            if len(group) == self.config.group_size:
                groups.append(group)
            
            group_idx += 1
        
        print(f"✅ 创建了 {len(groups)} 组数据")
        return groups


# ============================================================================
# GRPO训练器
# ============================================================================

class GRPOTrainer:
    """GRPO强化学习训练器（含训练稳定性机制）
    
    核心稳定性设计：
      1. Clip Ratio  — PPO-style 策略梯度裁剪，防止单步更新过激
      2. KL 惩罚     — 约束策略偏离 SFT 参考模型，防止灾难性遗忘
      3. 熵正则      — 鼓励输出多样性，防止熵坍缩（模型退化为固定模板）
      4. 熵告警      — 实时检测并警告熵坍缩/爆炸
    """
    
    def __init__(self, config: QwenGRPOConfig, model, tokenizer):
        self.config = config
        self.model = model          # 策略模型（训练中更新，可能是 DDP 包装后的对象）
        self.tokenizer = tokenizer
        self.reward_calculator = GRPORewardCalculator(config)

        # ----------------------------------------------------------------
        # optimizer 需绑定底层 module 参数（DDP wrapper 下 model.parameters()
        # 与 model.module.parameters() 等价，但显式获取 module 更稳健）
        # ----------------------------------------------------------------
        raw_model = model.module if isinstance(model, DDP) else model
        self.optimizer = AdamW(raw_model.parameters(), lr=config.learning_rate)

        # ----------------------------------------------------------------
        # 参考模型（Reference Model）：SFT 模型的冻结副本
        # 用途：计算 KL(π_θ || π_ref)，防止策略偏离 SFT 成果过远
        #
        # DDP 模式（同机多卡）：ref_model 放在与 policy model 相同的 GPU，
        #   避免 CPU↔GPU 数据搬运瓶颈。两卡各持有独立副本，互不干扰。
        # 单卡模式（显存受限）：ref_model 放 CPU，计算时临时移动到 GPU。
        # ----------------------------------------------------------------
        import copy
        self.ref_model = copy.deepcopy(raw_model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        is_ddp = dist.is_available() and dist.is_initialized()
        if is_ddp:
            # DDP 模式：ref_model 与 policy model 共享同一块 GPU
            self.ref_model = self.ref_model.to(raw_model.device)
            rank = dist.get_rank()
            print(f"[rank {rank}] ✅ 参考模型已放置到 GPU {raw_model.device}（DDP 模式）")
        else:
            # 单卡模式：ref_model offload 到 CPU，节省显存
            self.ref_model = self.ref_model.to('cpu')
            print("✅ 参考模型已 offload 到 CPU（单卡模式，节省显存）")
    
    # ================================================================
    # 训练稳定性核心方法
    # ================================================================

    def _compute_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_start_idx: int,
    ) -> torch.Tensor:
        """
        计算模型在 response 部分每个 token 上的 log probability。

        参数：
          model            — 策略模型或参考模型
          input_ids        — 完整序列（prompt + response），shape: [1, seq_len]
          attention_mask   — 对应的 attention mask
          response_start_idx — response 在序列中的起始位置

        返回：
          log_probs  — shape: [response_len]，仅包含 response 部分的 logprob
        """
        with torch.no_grad() if model is self.ref_model else torch.enable_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        # logits: [1, seq_len, vocab_size]
        logits = outputs.logits[0]                          # [seq_len, vocab_size]
        log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)  # [seq_len, vocab_size]

        # 取 response 部分（teacher-forcing：位置 t 预测 t+1）
        # response tokens: input_ids[response_start_idx:]
        # 对应预测 logits: log_probs_all[response_start_idx-1 : seq_len-1]
        resp_ids      = input_ids[0, response_start_idx:]              # [resp_len]
        resp_logprobs = log_probs_all[response_start_idx - 1: -1]      # [resp_len, vocab]

        # 取每个 token 对应 id 的 logprob
        per_token_logprob = resp_logprobs.gather(
            dim=-1, index=resp_ids.unsqueeze(-1)
        ).squeeze(-1)                                                   # [resp_len]

        return per_token_logprob

    def _compute_entropy(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        计算策略模型在 response 部分的平均 token 级分布熵 H(π_θ)。

        用途：
          1. 熵正则项（entropy_bonus）：loss -= entropy_coef * mean_entropy
          2. 熵坍缩/爆炸检测：实时告警

        返回：
          mean_entropy — scalar Tensor，当前 response 的平均 token 熵
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits[0]                                     # [seq_len, vocab]
        probs  = torch.nn.functional.softmax(logits, dim=-1)           # [seq_len, vocab]
        # H = -Σ p*log(p)，对 vocab 维度求和
        entropy_per_token = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)  # [seq_len]
        return entropy_per_token.mean()

    def _compute_kl_divergence(
        self,
        policy_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 token 级 KL 散度的序列均值 KL(π_θ || π_ref)。

        支持三种估计器（由 config.kl_estimator 控制）：

          k1 — 一阶近似（TRL/OpenRLHF 常用）：
               KL ≈ log(π_ref / π_θ) = ref_logprob - policy_logprob
               特点：低方差，有轻微正偏，计算极快

          k2 — 二阶对称近似（veRL/GRPO 推荐，当前默认）：
               令 r = exp(policy_logprob - ref_logprob) = π_θ/π_ref
               KL ≈ (r - 1)² / 2
               特点：无偏，方差低于 k3，r 在 1 附近时精度最优
               适合 GRPO 训练初中期（策略未大幅偏离参考模型时）

          k3 — 三阶无偏估计（DeepSeekMath/GRPO 论文）：
               令 r = exp(policy_logprob - ref_logprob) = π_θ/π_ref
               KL ≈ (r - 1) - log(r)
               特点：无偏估计，恒≥0，r 偏离较大时比 k2 更准确

        返回：
          kl_mean — scalar Tensor，序列平均 KL 值（用于 kl_penalty 项）
        """
        log_ratio = policy_logprobs - ref_logprobs      # log(r)，r = π_θ/π_ref
        ratio     = torch.exp(log_ratio)                  # r

        if self.config.kl_estimator == "k1":
            # k1: KL ≈ ref_logprob - policy_logprob（per token）
            kl_per_token = ref_logprobs - policy_logprobs
        elif self.config.kl_estimator == "k2":
            # k2: KL ≈ (r - 1)² / 2，二阶对称近似（veRL/GRPO 推荐）
            kl_per_token = (ratio - 1).pow(2) / 2
        else:
            # k3: KL ≈ (r-1) - log(r)，三阶无偏估计，恒 ≥ 0
            kl_per_token  = (ratio - 1) - log_ratio
        return kl_per_token.mean()

    def _compute_policy_gradient_loss(
        self,
        policy_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantage: float,
    ) -> Tuple[torch.Tensor, float]:
        """
        PPO-style 带 Clip Ratio 的策略梯度损失（Clipped Surrogate Objective）。

        公式：
          r_t     = exp(log π_θ(a_t) - log π_ref(a_t))   # 重要性权重（概率比）
          A_t     = advantage                              # 标量奖励优势值
          L_clip  = -mean[ min(r_t * A_t, clip(r_t, clip_low, clip_high) * A_t) ]
          clip_low  = 1 - clip_ratio      = 0.85
          clip_high = clip_ratio_high     = 0.28  ← veRL actor_rollout_ref.actor.clip_ratio_high

        非对称裁剪作用：
          r_t > 0.28：正激励方向梯度截断，防止奖励 hack 导致的激进更新。
          r_t < 0.85：负激励方向梯度截断，防止策略过度远离参考模型。
          非对称设计使正方向（鼓励）比负方向（惩罚）受到更严格的约束。

        参数：
          policy_logprobs — 策略模型的 per-token logprob [resp_len]
          ref_logprobs    — 参考模型的 per-token logprob [resp_len]
          advantage       — 该样本的优势值（reward - group_mean_reward）

        返回：
          (pg_loss, clip_frac)
            pg_loss   — 策略梯度损失（scalar Tensor，可直接 .backward()）
            clip_frac — 被裁剪的 token 比例（用于监控，越高说明更新越激进）
        """
        log_ratio = policy_logprobs - ref_logprobs          # log(π_θ/π_ref)
        ratio     = torch.exp(log_ratio)                    # π_θ/π_ref，重要性权重

        # 非对称裁剪区间：[clip_low, clip_high]
        # 对应 veRL: clip_ratio=0.15(下界偏移), clip_ratio_high=0.28(上界绝对值)
        clip_low  = 1.0 - self.config.clip_ratio       # 0.85
        clip_high = self.config.clip_ratio_high         # 0.28

        # 未裁剪的目标：r_t * A
        unclipped = ratio * advantage
        # 裁剪后的目标：clip(r_t, clip_low, clip_high) * A
        clipped   = torch.clamp(ratio, clip_low, clip_high) * advantage
        # 取最小值（保守更新）并取负号（转为最小化）
        pg_loss   = -torch.min(unclipped, clipped).mean()

        # 统计被裁剪的 token 比例（分别统计上下界触发情况）
        clip_frac = (
            (ratio < clip_low).float().mean().item() +
            (ratio > clip_high).float().mean().item()
        )

        return pg_loss, clip_frac

    def train_epoch(self, groups: List[List[Dict]]) -> Dict:
        """
        训练一个 epoch。

        损失函数由三项组成（在 GRPO 阶段应用所有稳定性机制）：

          total_loss = pg_loss          # ① PPO-style 裁剪策略梯度（主要学习信号）
                     + kl_coef * kl    # ② KL 散度惩罚（防止偏离 SFT 参考模型）
                     - entropy_coef * H # ③ 熵正则（防止熵坍缩，鼓励输出多样性）
        """
        self.model.train()

        # ----------------------------------------------------------------
        # DDP 兼容：获取底层 module 和设备
        #   DDP 包装后 self.model 是 DDP 对象，需通过 .module 访问真实模型
        #   .device 需从 module 的参数推断
        # ----------------------------------------------------------------
        is_ddp = dist.is_available() and dist.is_initialized()
        raw_model    = self.model.module if is_ddp else self.model
        policy_device = next(raw_model.parameters()).device
        is_main      = (not is_ddp) or (dist.get_rank() == 0)

        # epoch 级累积统计
        total_loss       = 0.0
        total_reward     = 0.0
        total_pg_loss    = 0.0
        total_kl         = 0.0
        total_entropy    = 0.0
        total_clip_frac  = 0.0
        entropy_alarms   = 0       # 触发熵告警的步数
        
        # 追踪难刻画字段的奖励变化趋势
        field_reward_accumulator: Dict[str, List[float]] = {
            "职业": [], "决策": [], "收入": [], "信用": [], "交易": [], "综合评分": []
        }
        
        # 进度条只在主进程输出，避免多进程日志混乱
        progress_bar = tqdm(groups, desc="Training", disable=not is_main)
        
        for group_idx, group in enumerate(progress_bar):
            
            # ----------------------------------------------------------------
            # Step 1：为组内每个样本生成预测文本（no_grad，仅用于奖励计算）
            # ----------------------------------------------------------------
            predictions = []
            for sample in group:
                prompt_inputs = self.tokenizer(
                    sample['prompt'],
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.config.max_seq_length,
                ).to(policy_device)
                with torch.no_grad():
                    # DDP 对象不支持直接调用 generate，需用底层 raw_model
                    gen_ids = raw_model.generate(
                        **prompt_inputs,
                        max_new_tokens=256,  # target 均值~80 tokens，256 足够且节省约一半生成时间
                        top_p=0.9,
                        temperature=0.7,
                        do_sample=True,
                    )
                pred_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                predictions.append(pred_text)
            
            targets = [s['target'] for s in group]
            
            # ----------------------------------------------------------------
            # Step 2：差异化奖励计算 + 计算组内优势值（advantage）
            #   advantage_i = reward_i - mean(rewards)
            #   使用相对优势而非绝对奖励，降低奖励方差
            # ----------------------------------------------------------------
            group_rewards = self.reward_calculator.calculate_group_reward(predictions, targets)
            mean_reward   = float(np.mean(group_rewards))
            advantages    = [r - mean_reward for r in group_rewards]
            
            # 收集字段级奖励详情
            for pred, tgt in zip(predictions, targets):
                field_details = self.reward_calculator.calculate_field_level_reward_detailed(pred, tgt)
                for field_name, score in field_details.items():
                    if field_name in field_reward_accumulator:
                        field_reward_accumulator[field_name].append(score)
            
            # ----------------------------------------------------------------
            # Step 3：对每个样本计算三项稳定性损失并累加
            # ----------------------------------------------------------------
            batch_pg_loss   = 0.0
            batch_kl        = 0.0
            batch_entropy   = 0.0
            batch_clip_frac = 0.0
            
            self.optimizer.zero_grad()
            
            for sample, advantage in zip(group, advantages):
                prompt   = sample['prompt']
                target   = sample['target']
                full_text = f"{prompt}\n\n评估结果：{target}"
                
                # 分词，获取 prompt 长度以定位 response 起始位置
                prompt_ids = self.tokenizer(
                    prompt, return_tensors='pt', truncation=True,
                    max_length=self.config.max_seq_length,
                )['input_ids']
                resp_start = prompt_ids.shape[1]  # response 在完整序列中的起始索引
                
                full_inputs = self.tokenizer(
                    full_text,
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    return_tensors='pt',
                ).to(policy_device)
                
                input_ids      = full_inputs['input_ids']
                attention_mask = full_inputs['attention_mask']
                
                # 安全检查：确保 response 部分有足够长度
                if resp_start >= input_ids.shape[1] - 1:
                    continue
                
                # ─── policy 单次前向，同时取 logprobs + entropy（避免两次重复计算）───
                policy_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                policy_logits = policy_outputs.logits[0]          # [seq, vocab]
                log_probs_all = torch.nn.functional.log_softmax(policy_logits, dim=-1)
                probs_all     = torch.exp(log_probs_all)

                # policy logprobs（response 部分，带梯度）
                resp_ids        = input_ids[0, resp_start:]
                resp_logprobs   = log_probs_all[resp_start - 1: -1]
                policy_logprobs = resp_logprobs.gather(
                    dim=-1, index=resp_ids.unsqueeze(-1)
                ).squeeze(-1)                                      # [resp_len]

                # entropy（全序列均值，仍带梯度用于 entropy_coef 项）
                entropy = -(probs_all * torch.log(probs_all + 1e-9)).sum(dim=-1).mean()

                # ② 计算 ref logprobs（no_grad，参考模型冻结）
                ref_inputs = {k: v.to(self.ref_model.device) for k, v in full_inputs.items()}
                ref_logprobs = self._compute_log_probs(
                    self.ref_model,
                    ref_inputs['input_ids'],
                    ref_inputs['attention_mask'],
                    resp_start,
                )
                ref_logprobs = ref_logprobs.to(policy_logprobs.device)

                # ① PPO-style 裁剪策略梯度损失
                pg_loss, clip_frac = self._compute_policy_gradient_loss(
                    policy_logprobs, ref_logprobs, advantage
                )

                # ② KL 散度惩罚（防止策略偏离 SFT 参考模型）
                kl = self._compute_kl_divergence(policy_logprobs, ref_logprobs)
                
                # ----------------------------------------------------------------
                # 熵告警：实时检测熵坍缩/爆炸
                # ----------------------------------------------------------------
                entropy_val = entropy.item()
                if entropy_val < self.config.entropy_low_threshold:
                    print(f"\n⚠️  [熵坍缩警告] step={group_idx} entropy={entropy_val:.4f} "
                          f"< threshold={self.config.entropy_low_threshold}  "
                          f"模型输出趋于固定模板，考虑增大 entropy_coef")
                    entropy_alarms += 1
                elif entropy_val > self.config.entropy_high_threshold:
                    print(f"\n⚠️  [熵爆炸警告] step={group_idx} entropy={entropy_val:.4f} "
                          f"> threshold={self.config.entropy_high_threshold}  "
                          f"模型输出趋于混乱，考虑减小 entropy_coef 或增大 kl_coef")
                    entropy_alarms += 1
                
                # 三项组合损失
                loss = (
                    pg_loss
                    + self.config.kl_coef     * kl
                    - self.config.entropy_coef * entropy   # 最大化熵 → 最小化负熵
                )
                loss.backward()
                
                batch_pg_loss   += pg_loss.item()
                batch_kl        += kl.item()
                batch_entropy   += entropy_val
                batch_clip_frac += clip_frac
            
            n_samples = len(group)
            batch_pg_loss   /= n_samples
            batch_kl        /= n_samples
            batch_entropy   /= n_samples
            batch_clip_frac /= n_samples
            
            # 梯度裁剪（使用更保守的 max_grad_norm=0.5）
            # DDP 模式下梯度已经由各进程自动 all-reduce，裁剪针对底层 module 参数
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            # 累积统计
            total_loss      += batch_pg_loss + self.config.kl_coef * batch_kl
            total_reward    += mean_reward
            total_pg_loss   += batch_pg_loss
            total_kl        += batch_kl
            total_entropy   += batch_entropy
            total_clip_frac += batch_clip_frac
            
            # ----------------------------------------------------------------
            # 进度条：同时展示奖励、稳定性指标（熵/KL/裁剪率）、难字段奖励
            # ----------------------------------------------------------------
            job_reward_recent = (
                np.mean(field_reward_accumulator["职业"][-20:])
                if field_reward_accumulator["职业"] else 0.0
            )
            dec_reward_recent = (
                np.mean(field_reward_accumulator["决策"][-20:])
                if field_reward_accumulator["决策"] else 0.0
            )
            progress_bar.set_postfix({
                'loss':    f"{batch_pg_loss:.4f}",
                'reward':  f"{mean_reward:.4f}",
                'H':       f"{batch_entropy:.2f}",      # 熵（正常范围 0.5~8.0）
                'kl':      f"{batch_kl:.4f}",           # KL 散度（越小越保守）
                'clip%':   f"{batch_clip_frac*100:.1f}", # 被裁剪的 token 百分比
                'job_r':   f"{job_reward_recent:.3f}",
                'dec_r':   f"{dec_reward_recent:.3f}",
            })
        
        n_groups = len(groups)
        field_avg_rewards = {
            field: float(np.mean(scores)) if scores else 0.0
            for field, scores in field_reward_accumulator.items()
        }
        
        return {
            'avg_loss':     total_loss      / n_groups,
            'avg_reward':   total_reward    / n_groups,
            # 稳定性指标
            'avg_pg_loss':  total_pg_loss   / n_groups,
            'avg_kl':       total_kl        / n_groups,
            'avg_entropy':  total_entropy   / n_groups,
            'avg_clip_frac': total_clip_frac / n_groups,
            'entropy_alarm_steps': entropy_alarms,
            # 字段级奖励统计
            'field_rewards': field_avg_rewards,
            'hard_field_job_reward':      field_avg_rewards.get("职业", 0.0),
            'hard_field_decision_reward': field_avg_rewards.get("决策", 0.0),
            'easy_field_avg_reward': float(np.mean([
                field_avg_rewards.get(f, 0.0) for f in ["收入", "信用", "交易"]
            ])),
        }
    
    def train(self, data: List[Dict]):
        """执行GRPO训练"""
        
        print("=" * 80)
        print("🚀 Qwen2-7B-Instruct KYC GRPO强化学习训练")
        print("=" * 80)
        
        processor = GRPODataProcessor(self.config, self.tokenizer)
        groups = processor.create_groups(data)
        
        for epoch in range(self.config.num_train_epochs):
            print(f"\n【Epoch {epoch + 1}/{self.config.num_train_epochs}】")
            
            metrics = self.train_epoch(groups)
            
            print(f"  平均损失:     {metrics['avg_loss']:.4f}")
            print(f"  平均奖励:     {metrics['avg_reward']:.4f}")
            print(f"  --- 训练稳定性指标 ---")
            print(f"  策略梯度损失: {metrics['avg_pg_loss']:.4f}")
            print(f"  KL 散度:      {metrics['avg_kl']:.4f}  (估计器: {self.config.kl_estimator}, 系数: {self.config.kl_coef})")
            print(f"  策略熵 H:     {metrics['avg_entropy']:.4f}  "
                  f"(正常范围 {self.config.entropy_low_threshold:.1f}~{self.config.entropy_high_threshold:.1f})"
                  + ("  ⚠️ 告警!" if metrics['entropy_alarm_steps'] > 0 else "  ✅ 正常"))
            print(f"  Clip 比例:    {metrics['avg_clip_frac']*100:.1f}%  "
                  f"(区间 [{1.0-self.config.clip_ratio:.2f}, {self.config.clip_ratio_high:.2f}]，建议 <30% 为健康)")
            print(f"  熵告警次数:   {metrics['entropy_alarm_steps']} 步")
            print(f"  --- 字段级奖励详情（差异化激励监控）---")
            print(f"  [难刻画] 职业字段奖励: {metrics['hard_field_job_reward']:.4f}  (权重x2.5)")
            print(f"  [难刻画] 决策字段奖励: {metrics['hard_field_decision_reward']:.4f}  (权重x2.0)")
            print(f"  [常规]   收入/信用/交易均值: {metrics['easy_field_avg_reward']:.4f}  (权重x1.0)")
            
            # wandb 已禁用，指标已通过 print 输出到本地终端
        
        # 保存模型：DDP 模式下只有 rank-0 主进程执行保存，避免多进程重复写文件
        is_ddp  = dist.is_available() and dist.is_initialized()
        is_main = (not is_ddp) or (dist.get_rank() == 0)
        if is_main:
            raw_model = self.model.module if is_ddp else self.model
            print(f"\n💾 保存模型到: {self.config.output_dir}")
            raw_model.save_pretrained(self.config.output_dir)
            self.tokenizer.save_pretrained(self.config.output_dir)
        if is_ddp:
            dist.barrier()   # 等待主进程保存完毕，其他进程再继续
        

# ============================================================================
# 每个 DDP 进程的训练入口
# ============================================================================

def train_worker(rank: int, world_size: int, sft_model_path: str, data_path: str):
    """
    单个训练进程的完整逻辑。
    由 mp.spawn 拉起（双卡）或直接调用（单卡）。

    参数：
      rank           — 当前进程编号（0 或 1）
      world_size     — 总进程数（双卡=2，单卡=1）
      sft_model_path — SFT 模型路径
      data_path      — 训练数据路径
    """
    use_ddp = world_size > 1
    is_main = (rank == 0)

    # ─────────────────────────────────────────────────────────────────────────
    # DDP 进程组初始化
    #   使用 gloo backend 作为备选（NCCL 在部分 AutoDL 环境有段错误），
    #   若 NCCL 可用则优先用 NCCL，否则回退 gloo（gloo 性能略低但兼容性好）
    # ─────────────────────────────────────────────────────────────────────────
    if use_ddp:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        # 优先尝试 NCCL，失败则回退 gloo
        try:
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
            )
            backend_used = "nccl"
        except Exception as e:
            print(f"[rank {rank}] NCCL 初始化失败 ({e})，回退到 gloo")
            dist.init_process_group(
                backend="gloo",
                rank=rank,
                world_size=world_size,
            )
            backend_used = "gloo"
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        if is_main:
            print(f"🚀 DDP 初始化完成（{backend_used}）：{world_size} 张 GPU")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if is_main:
            print(f"▶  单卡模式，使用设备: {device}")

    # ─────────────────────────────────────────────────────────────────────────
    # GRPO 配置
    # ─────────────────────────────────────────────────────────────────────────
    config = QwenGRPOConfig(
        model_name=sft_model_path,
        sft_model_path=sft_model_path,  # 这里应该是合并后的模型路径
        data_path=data_path,
        num_train_epochs=3,
        group_size=4,
        learning_rate=5e-6,
        output_dir="./qwen2_kyc_grpo_model",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 加载模型
    #   主进程 (rank 0) 已在 __main__ 中完成 LoRA 合并到 SFT_MODEL_PATH
    #   DDP 模式下各进程各自加载一份，无跨 GPU 数据转移
    # ─────────────────────────────────────────────────────────────────────────
    if is_main:
        print(f"📦 加载 SFT 模型作为 GRPO 策略模型起点: {sft_model_path}")
    
    # 验证模型路径是否存在
    sft_path = Path(sft_model_path)
    if not sft_path.exists():
        print(f"❌ 错误: SFT 模型路径不存在: {sft_model_path}")
        exit(1)
    
    # 检查是否是 LoRA adapter 或完整模型
    config_path = sft_path / "config.json"
    adapter_config_path = sft_path / "adapter_config.json"
    
    is_lora = adapter_config_path.exists()
    is_complete = config_path.exists()
    
    if not is_complete and not is_lora:
        print(f"❌ 错误: {sft_model_path} 既不是完整模型也不是 LoRA adapter")
        print(f"   缺少 config.json 或 adapter_config.json")
        exit(1)
    
    if is_lora and not is_complete:
        print(f"⚠️  检测到 LoRA adapter（未合并）")
        print(f"   注意：LoRA adapter 需要与基座模型合并才能正常使用")
        print(f"   如果网络不可用，训练效果可能会受影响")
    
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
    
    # 4bit 量化配置（大幅节省显存）
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device} if use_ddp else "auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
     
    )
    
    # 启用梯度检查点
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    # 清空缓存，为 DDP 初始化预留显存
    torch.cuda.empty_cache()
    if is_main:
        print("✅ 梯度检查点已启用，缓存已清空")

    # DDP 包装（设置 find_unused_parameters=True 避免显存不足时的问题）
    if use_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True, static_graph=True)
        if is_main:
            print(f"✅ 策略模型已用 DDP 包装（{world_size} 张 GPU）")

    # ─────────────────────────────────────────────────────────────────────────
    # 加载并分片数据
    # ─────────────────────────────────────────────────────────────────────────
    processor = GRPODataProcessor(config, tokenizer)
    data = processor.load_data()

    if use_ddp:
        shard_size = len(data) // world_size
        start_idx  = rank * shard_size
        data       = data[start_idx: start_idx + shard_size]
        print(f"[rank {rank}] 数据分片: [{start_idx}, {start_idx + shard_size})，共 {shard_size} 条")

    # 执行 GRPO 训练
    trainer = GRPOTrainer(config, model, tokenizer)
    trainer.train(data)

    if is_main:
        print("\n" + "=" * 80)
        print("✅ GRPO 训练完成！")
        print("=" * 80)
        print(f"\n💾 GRPO 模型已保存到: {config.output_dir}")
        print(f"\n📋 两阶段训练结果：")
        print(f"   SFT  模型: {sft_model_path}")
        print(f"   GRPO 模型: {config.output_dir}  ← 推荐用于生产部署")
        print(f"\n🔍 验证效果:")
        print(f"   python qwen2_inference.py --model-path {config.output_dir} --mode interactive")

    if use_ddp:
        dist.destroy_process_group()


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    import torch.multiprocessing as mp

    # =========================================================================
    # 两阶段训练流程：SFT → GRPO
    #
    #   阶段一（已完成）：python qwen2_sft_trainer.py
    #     └─ 输出: ./qwen2_kyc_model   ← SFT 微调后的模型
    #
    #   阶段二（本脚本）：python qwen2_grpo_trainer.py
    #     └─ 加载 SFT 模型，在差异化奖励信号下继续强化学习
    #     └─ 输出: ./qwen2_kyc_grpo_model
    #
    # ─────────────────────────────────────────────────────────────────────────
    # 启动方式（只需一条命令，无需 torchrun）：
    #
    #   ① 单卡：
    #       python qwen2_grpo_trainer.py
    #
    #   ② 同机双卡（自动用 mp.spawn 拉起 2 个进程，不依赖 torchrun）：
    #       NUM_GPUS=2 python qwen2_grpo_trainer.py
    #       # 或直接在脚本里把下面的 num_gpus 改为 2
    # =========================================================================

    # LoRA adapter 目录（SFT 输出）
    SFT_ADAPTER_PATH = "./qwen2_kyc_model"
    # 合并后的完整模型目录（放数据盘，供 GRPO 加载）
    SFT_MODEL_PATH   = "/root/autodl-tmp/customer-analyse/qwen2_kyc_model_merged"
    DATA_PATH        = "/root/autodl-tmp/customer-analyse/financial_data/kyc_gspo_training_data_1_10.jsonl"

    # ─────────────────────────────────────────────────────────────────────────
    # 自动检测：SFT 输出是 LoRA adapter 还是完整模型
    #   LoRA 输出标志：目录下存在 adapter_config.json
    #   完整模型标志：目录下存在 config.json（无 adapter_config.json）
    # ─────────────────────────────────────────────────────────────────────────
    adapter_dir  = Path(SFT_ADAPTER_PATH)
    merged_dir   = Path(SFT_MODEL_PATH)
    is_lora_adapter = (adapter_dir / "adapter_config.json").exists()
    is_full_model   = (adapter_dir / "config.json").exists() and not is_lora_adapter

    if merged_dir.exists() and (merged_dir / "config.json").exists():
        # 已经合并过，直接使用
        print(f"✅ 检测到已合并的完整模型: {SFT_MODEL_PATH}，跳过合并步骤")

    elif is_lora_adapter:
        # SFT 输出是 LoRA adapter，需要合并到本地基座模型
        print("🔀 检测到 LoRA adapter，开始合并到本地基座模型...")
        print(f"   adapter 路径: {SFT_ADAPTER_PATH}")
        print(f"   合并输出路径: {SFT_MODEL_PATH}")

        try:
            from peft import PeftModel
        except ImportError:
            print("❌ 缺少 peft 库，请安装: pip install peft")
            exit(1)
        base_model_path = "./models/Qwen2-7B-Instruct"
        
        # 验证模型存在性
        if not Path(base_model_path).exists():
            print(f"❌ 错误：未找到本地基座模型: {base_model_path}")
            print("   请确保基座模型已下载到 ./models 目录中")
            exit(1)


        print(f"   基座模型: {base_model_path}")
        print("   加载基座模型（约 14GB，需要 1~2 分钟）...")
        
        # 离线模式：如果是本地路径但不存在，给出错误提示
        if not base_model_path.startswith("http") and not Path(base_model_path).exists():
            print(f"❌ 错误：基座模型路径不存在: {base_model_path}")
            print()
            print("解决方案：")
            print("1. 检查网络连接，脚本会自动下载基座模型")
            print("2. 或者手动下载模型到本地缓存：")
            print("   python3 << 'EOF'")
            print("   from transformers import AutoModel")
            print("   AutoModel.from_pretrained('Qwen/Qwen2-7B-Instruct')")
            print("   EOF")
            exit(1)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

        print("   加载 LoRA adapter 并合并权重...")
        peft_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
        merged_model = peft_model.merge_and_unload()

        print(f"   保存完整模型到: {SFT_MODEL_PATH}")
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(SFT_MODEL_PATH)
        tokenizer.save_pretrained(SFT_MODEL_PATH)

        del base_model, peft_model, merged_model
        torch.cuda.empty_cache()
        print("✅ LoRA 合并完成！")

    elif is_full_model:
        # 直接是完整模型，路径对齐
        SFT_MODEL_PATH = SFT_ADAPTER_PATH
        print(f"✅ 检测到完整 SFT 模型: {SFT_MODEL_PATH}，直接使用")

    else:
        print(f"❌ 未找到有效的 SFT 模型或 LoRA adapter: {SFT_ADAPTER_PATH}")
        print(f"   请先运行: python qwen2_sft_trainer.py")
        exit(1)
    
    # 检查是否启用离线模式
    use_offline = os.environ.get("OFFLINE", "0") == "1"
    if use_offline and is_lora_adapter:
        print()
        print("⚠️  离线模式：将跳过 LoRA 合并，直接使用 LoRA adapter")
        print("   （仅适用于推理，训练效果会受影响）")
        SFT_MODEL_PATH = SFT_ADAPTER_PATH

    print(f"✅ SFT 模型就绪: {SFT_MODEL_PATH}，开始 GRPO 阶段训练")

    # ─────────────────────────────────────────────────────────────────────────
    # 决定使用几张 GPU
    #   - 优先读环境变量 NUM_GPUS（方便命令行覆盖）
    #   - 其次自动检测可用 GPU 数量
    #   - 单卡/无 GPU 时退化为 world_size=1
    # ─────────────────────────────────────────────────────────────────────────
    available_gpus = torch.cuda.device_count()
    num_gpus = int(os.environ.get("NUM_GPUS", available_gpus))
    num_gpus = max(1, min(num_gpus, available_gpus))   # 不超过实际可用卡数
    print(f"🖥️  检测到 {available_gpus} 张 GPU，本次使用 {num_gpus} 张")

    # 默认启用 DDP 多卡训练（通过环境变量 ENABLE_DDP=0 可禁用）
    # 多卡训练可以有效降低单卡显存压力
    enable_ddp = os.environ.get("ENABLE_DDP", "1") == "1"  # 默认启用
    
    if enable_ddp and num_gpus > 1:
        print(f"✅ 启用 DDP 多进程模式，使用 {num_gpus} 张 GPU")
        # mp.spawn 在同一进程内 fork 出 num_gpus 个子进程，
        # 完全绕开 torchrun 的 rendezvous 机制，规避 c10d 段错误
        mp.spawn(
            train_worker,
            args=(num_gpus, SFT_MODEL_PATH, DATA_PATH),
            nprocs=num_gpus,
            join=True,
        )
    else:
        # 单卡直接调用，无多进程开销
        print("✅ 使用单进程模式（稳定可靠）")
        train_worker(rank=0, world_size=1,
                     sft_model_path=SFT_MODEL_PATH, data_path=DATA_PATH)
