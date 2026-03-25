"""
Qwen2 KYC模型评估脚本

评估指标:
  - BLEU分数 (文本相似度)
  - ROUGE分数 (摘要质量)
  - 风险评估准确率 (主任务)
  - 推理链路完整性
  - 决策准确性
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 评估函数
# ============================================================================

class KYCModelEvaluator:
    """KYC模型评估器"""
    
    def __init__(self, model_path: str, device='cuda'):
        """初始化评估器"""
        
        print(f"📦 加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.device = device
        self.model.eval()
        
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    def load_test_data(self, data_path: str) -> List[Dict]:
        """加载测试数据"""
        print(f"📥 加载测试数据: {data_path}")
        
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        print(f"✅ 已加载 {len(data):,} 条记录")
        return data
    
    def generate_prediction(self, kyc_text: str, max_length: int = 1024) -> str:
        """为KYC文本生成预测"""
        
        prompt = f"请基于以下KYC材料，完成多步骤的风险评估分析：\n\n{kyc_text}\n\n评估结果："
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                top_p=0.9,
                temperature=0.7,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取评估部分
        if "评估结果：" in prediction:
            prediction = prediction.split("评估结果：")[-1]
        
        return prediction.strip()
    
    def calculate_bleu_score(self, prediction: str, target: str) -> float:
        """计算BLEU分数"""
        pred_tokens = prediction.split()
        target_tokens = target.split()
        
        smoothing = SmoothingFunction().method1
        bleu = sentence_bleu(
            [target_tokens],
            pred_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        )
        
        return bleu
    
    def calculate_rouge_score(self, prediction: str, target: str) -> Dict[str, float]:
        """计算ROUGE分数"""
        scores = self.rouge_scorer.score(target, prediction)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure,
        }
    
    def extract_reasoning_steps(self, text: str) -> List[str]:
        """提取推理步骤"""
        steps = []
        parts = text.split('|')
        for part in parts:
            if '[' in part and ']' in part:
                steps.append(part.strip())
        return steps
    
    def calculate_reasoning_completeness(self, prediction: str, target: str) -> float:
        """
        计算推理完整性
        检查是否包含所有必需的推理步骤
        """
        pred_steps = self.extract_reasoning_steps(prediction)
        target_steps = self.extract_reasoning_steps(target)
        
        if len(target_steps) == 0:
            return 0.0
        
        # 检查步骤覆盖率
        coverage = min(len(pred_steps), len(target_steps)) / len(target_steps)
        
        return coverage
    
    def extract_risk_label(self, text: str) -> int:
        """从文本中提取风险标签 (-1: 无法解析, 0: 低风险, 1: 高风险)"""
        
        if '低风险' in text or 'low risk' in text.lower():
            return 0
        elif '高风险' in text or 'high risk' in text.lower():
            return 1
        else:
            return -1
    
    def calculate_risk_assessment_accuracy(self, predictions: List[str], targets: List[str], true_labels: List[int]) -> Dict[str, float]:
        """
        计算风险评估准确率
        """
        
        pred_labels = [self.extract_risk_label(p) for p in predictions]
        target_labels = [self.extract_risk_label(t) for t in targets]
        
        # 过滤掉无法解析的标签
        valid_indices = [i for i, (pl, tl) in enumerate(zip(pred_labels, target_labels)) if pl != -1 and tl != -1]
        
        if len(valid_indices) == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'valid_count': 0,
            }
        
        valid_pred = [pred_labels[i] for i in valid_indices]
        valid_target = [target_labels[i] for i in valid_indices]
        
        # 计算准确率
        correct = sum(1 for p, t in zip(valid_pred, valid_target) if p == t)
        accuracy = correct / len(valid_indices)
        
        # 计算精准率和召回率
        tp = sum(1 for p, t in zip(valid_pred, valid_target) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(valid_pred, valid_target) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(valid_pred, valid_target) if p == 0 and t == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'valid_count': len(valid_indices),
        }
    
    def evaluate(self, test_data: List[Dict]) -> Dict:
        """
        进行完整的评估
        """
        
        print("\n" + "=" * 80)
        print("🔍 开始评估...")
        print("=" * 80)
        
        all_predictions = []
        all_targets = []
        all_kyc_texts = []
        all_bleu_scores = []
        all_rouge_scores = {'rouge1': [], 'rougeL': []}
        all_reasoning_completeness = []
        all_true_labels = []
        
        progress_bar = tqdm(test_data, desc="Evaluating")
        
        for sample in progress_bar:
            kyc_text = sample['prompt']
            target = sample['target']
            true_label = sample['risk_label']
            
            # 生成预测
            prediction = self.generate_prediction(kyc_text)
            
            all_predictions.append(prediction)
            all_targets.append(target)
            all_kyc_texts.append(kyc_text)
            all_true_labels.append(true_label)
            
            # 计算指标
            bleu = self.calculate_bleu_score(prediction, target)
            all_bleu_scores.append(bleu)
            
            rouge = self.calculate_rouge_score(prediction, target)
            all_rouge_scores['rouge1'].append(rouge['rouge1'])
            all_rouge_scores['rougeL'].append(rouge['rougeL'])
            
            completeness = self.calculate_reasoning_completeness(prediction, target)
            all_reasoning_completeness.append(completeness)
        
        # 风险评估准确率
        risk_metrics = self.calculate_risk_assessment_accuracy(
            all_predictions,
            all_targets,
            all_true_labels
        )
        
        # 综合结果
        results = {
            'bleu_score': np.mean(all_bleu_scores),
            'rouge1_score': np.mean(all_rouge_scores['rouge1']),
            'rougeL_score': np.mean(all_rouge_scores['rougeL']),
            'reasoning_completeness': np.mean(all_reasoning_completeness),
            'risk_assessment_accuracy': risk_metrics['accuracy'],
            'risk_assessment_precision': risk_metrics['precision'],
            'risk_assessment_recall': risk_metrics['recall'],
            'risk_assessment_f1': risk_metrics['f1'],
            'valid_samples': risk_metrics['valid_count'],
            'total_samples': len(test_data),
        }
        
        return results, all_predictions, all_targets, all_kyc_texts
    
    def print_results(self, results: Dict):
        """打印评估结果"""
        
        print("\n" + "=" * 80)
        print("📊 评估结果")
        print("=" * 80)
        
        print("\n【文本相似度指标】")
        print(f"  BLEU分数:         {results['bleu_score']:.4f}")
        print(f"  ROUGE-1分数:      {results['rouge1_score']:.4f}")
        print(f"  ROUGE-L分数:      {results['rougeL_score']:.4f}")
        
        print("\n【推理链路指标】")
        print(f"  推理完整性:       {results['reasoning_completeness']:.4f}")
        
        print("\n【风险评估指标】")
        print(f"  准确率:           {results['risk_assessment_accuracy']:.4f}")
        print(f"  精准率:           {results['risk_assessment_precision']:.4f}")
        print(f"  召回率:           {results['risk_assessment_recall']:.4f}")
        print(f"  F1分数:           {results['risk_assessment_f1']:.4f}")
        print(f"  有效样本:         {results['valid_samples']}/{results['total_samples']}")
        
        # 综合评分
        composite_score = (
            results['bleu_score'] * 0.15 +
            results['rouge1_score'] * 0.15 +
            results['reasoning_completeness'] * 0.20 +
            results['risk_assessment_accuracy'] * 0.50
        )
        
        print(f"\n【综合评分】")
        print(f"  总体得分:         {composite_score:.4f}/1.0")
        
        if composite_score >= 0.8:
            print(f"  评价:             优秀 ⭐⭐⭐⭐⭐")
        elif composite_score >= 0.7:
            print(f"  评价:             良好 ⭐⭐⭐⭐")
        elif composite_score >= 0.6:
            print(f"  评价:             中等 ⭐⭐⭐")
        else:
            print(f"  评价:             需改进 ⭐⭐")
        
        print("=" * 80)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='./qwen2_kyc_model', help='模型路径')
    parser.add_argument('--test-data', default='/Applications/financial LLM/financial_data/kyc_gspo_training_data.jsonl', help='测试数据路径')
    parser.add_argument('--num-samples', type=int, default=100, help='评估样本数')
    parser.add_argument('--output-path', default='./evaluation_results.json', help='结果保存路径')
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = KYCModelEvaluator(args.model_path)
    
    # 加载测试数据
    test_data = evaluator.load_test_data(args.test_data)
    
    # 限制样本数
    if len(test_data) > args.num_samples:
        test_data = test_data[:args.num_samples]
        print(f"📌 限制评估样本数: {args.num_samples}")
    
    # 执行评估
    results, predictions, targets, kyc_texts = evaluator.evaluate(test_data)
    
    # 打印结果
    evaluator.print_results(results)
    
    # 保存详细结果
    print(f"\n💾 保存详细结果到: {args.output_path}")
    
    detailed_results = {
        'metrics': results,
        'samples': [
            {
                'kyc_text': text,
                'prediction': pred,
                'target': target,
            }
            for text, pred, target in zip(kyc_texts, predictions, targets)
        ]
    }
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    print("✅ 评估完成!")
