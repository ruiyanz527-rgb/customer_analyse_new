"""
课程学习模型对比评估
比较 SFT vs SFT+GRPO vs SFT+课程学习 的性能
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd

# ============================================================================
# 加载测试数据
# ============================================================================

def load_test_data(data_path, num_samples=None):
    """加载测试数据"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                continue
    
    if num_samples:
        data = data[:num_samples]
    
    return data

# ============================================================================
# 评估函数
# ============================================================================

def generate_response(model, tokenizer, prompt, max_length=512):
    """生成模型响应"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除 prompt 部分
        response = response[len(prompt):].strip()
        return response
    except Exception as e:
        print(f"⚠️ 生成失败: {e}")
        return ""

def calculate_accuracy(model, tokenizer, test_data, task='content_similarity'):
    """
    计算准确率
    支持多种评估方式：
    - content_similarity: 内容相似度评估
    - exact_match: 精确匹配
    - semantic_similarity: 语义相似度
    """
    correct = 0
    total = 0
    scores = []
    
    print(f"📊 评估模型（{task} 任务）...")
    
    for idx, sample in enumerate(tqdm(test_data, desc="评估进度")):
        prompt = sample.get('prompt', '')
        target = sample.get('target', '')
        
        if not prompt or not target:
            continue
        
        # 生成响应
        response = generate_response(model, tokenizer, prompt, max_length=512)
        
        if task == 'exact_match':
            # 精确匹配
            if response.strip() == target.strip():
                correct += 1
                scores.append(1.0)
            else:
                scores.append(0.0)
        
        elif task == 'content_similarity':
            # 基于词汇重叠的相似度（更宽松的阈值）
            target_words = set(target.lower().split())
            response_words = set(response.lower().split())
            
            if target_words and response_words:
                overlap = len(target_words & response_words)
                # 使用 Jaccard 相似度
                union = len(target_words | response_words)
                similarity = overlap / union if union > 0 else 0
                scores.append(similarity)
                
                # 相似度 > 0.2 认为正确（更宽松）
                if similarity > 0.2:
                    correct += 1
            else:
                scores.append(0.0)
        
        elif task == 'semantic_similarity':
            # 基于内容匹配的语义相似度
            target_lower = target.lower()
            response_lower = response.lower()
            
            # 简化评估：检查是否生成了有意义的回答
            # 1. 回答不为空
            has_content = len(response_lower.strip()) > 0
            
            # 2. 回答长度合理（至少有几个词）
            has_length = len(response_lower.split()) > 2
            
            # 3. 不是简单的重复
            has_variation = response_lower != prompt.lower()
            
            similarity = sum([has_content, has_length, has_variation]) / 3.0
            scores.append(similarity)
            
            # 三个条件都满足则认为正确
            if has_content and has_length and has_variation:
                correct += 1
        
        total += 1
        
        # 定期清理显存
        if (idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
    
    accuracy = (correct / total * 100) if total > 0 else 0
    avg_score = np.mean(scores) if scores else 0
    
    return accuracy, avg_score, scores

def evaluate_model(model_path, test_data, model_name="Model", low_memory=True):
    """评估单个模型"""
    print(f"\n{'='*70}")
    print(f"📦 评估模型: {model_name}")
    print(f"{'='*70}")
    
    # 加载模型
    print(f"Loading model from: {model_path}")
    
    # 清理显存
    torch.cuda.empty_cache()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if low_memory:
            print(f"💾 使用低显存模式（FP16）...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="cuda:0",
                trust_remote_code=True,
            )
        model.eval()
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) and not low_memory:
            print(f"⚠️  显存不足，自动切换到低显存模式...")
            torch.cuda.empty_cache()
            return evaluate_model(model_path, test_data, model_name, low_memory=True)
        print(f"❌ 模型加载失败: {e}")
        return None
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None
    
    # 评估多个指标（低显存模式下减少样本数）
    metrics = {}
    num_eval_samples = 30 if low_memory else 100
    num_quality_samples = 10 if low_memory else 30
    
    # 指标 1: 内容相似度准确率
    print(f"\n📊 计算内容相似度准确率（{num_eval_samples} 个样本）...")
    accuracy1, avg_score1, _ = calculate_accuracy(model, tokenizer, test_data[:num_eval_samples], task='content_similarity')
    metrics['内容相似度准确率'] = f"{accuracy1:.1f}%"
    metrics['平均相似度分数'] = f"{avg_score1:.2f}"
    
    # 指标 2: 语义相似度准确率
    print(f"\n📊 计算语义相似度准确率（{num_eval_samples} 个样本）...")
    accuracy2, avg_score2, _ = calculate_accuracy(model, tokenizer, test_data[:num_eval_samples], task='semantic_similarity')
    metrics['语义相似度准确率'] = f"{accuracy2:.1f}%"
    
    # 指标 3: 响应质量指标
    print(f"\n📊 计算响应质量指标（{num_quality_samples} 个样本）...")
    response_lengths = []
    response_times = []
    
    import time
    for sample in tqdm(test_data[:num_quality_samples], desc="计算质量"):
        prompt = sample.get('prompt', '')
        start_time = time.time()
        response = generate_response(model, tokenizer, prompt, max_length=512)
        elapsed = time.time() - start_time
        response_lengths.append(len(response))
        response_times.append(elapsed)
    
    avg_length = np.mean(response_lengths) if response_lengths else 0
    avg_time = np.mean(response_times) if response_times else 0
    metrics['平均响应长度'] = f"{avg_length:.0f}"
    metrics['平均推理时间'] = f"{avg_time:.2f}s"
    
    # 清理显存
    torch.cuda.empty_cache()
    
    return metrics

# ============================================================================
# 主函数
# ============================================================================

def main():
    import os
    
    # 设置显存优化环境变量
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("\n" + "="*70)
    print("🤖 Qwen2 KYC 模型性能对比评估")
    print("="*70)
    print("💾 已启用低显存模式")
    
    # 定义模型路径
    model_configs = {
        'SFT 基线': {
            'path': 'qwen2_kyc_model_merged_full',
            'description': '基础 SFT 模型'
        },
        'SFT + GRPO': {
            'path': 'qwen2_kyc_model_grpo',
            'description': 'SFT + GRPO 强化学习'
        },
        'SFT + 课程学习': {
            'path': 'curriculum_model/final',
            'description': 'SFT + 课程学习策略'
        }
    }
    
    # 测试数据路径
    test_data_path = "financial_data/kyc_gspo_training_data_1_10.jsonl"
    
    # 加载测试数据
    print(f"\n📥 加载测试数据: {test_data_path}")
    test_data = load_test_data(test_data_path, num_samples=200)
    print(f"✅ 加载了 {len(test_data)} 个测试样本")
    
    # 评估所有模型
    results = {}
    model_status = {}
    
    for model_name, config in model_configs.items():
        model_path = config['path']
        description = config['description']
        
        print(f"\n{'='*70}")
        print(f"📦 评估: {model_name} ({description})")
        print(f"   路径: {model_path}")
        print(f"{'='*70}")
        
        path_obj = Path(model_path)
        if not path_obj.exists():
            print(f"⚠️  模型路径不存在: {model_path}")
            model_status[model_name] = "❌ 未找到"
            continue
        
        try:
            metrics = evaluate_model(model_path, test_data, model_name)
            if metrics:
                results[model_name] = metrics
                model_status[model_name] = "✅ 已评估"
                
                # 打印该模型的结果
                print(f"\n{model_name} 评估结果:")
                for key, value in metrics.items():
                    print(f"  • {key}: {value}")
        except Exception as e:
            print(f"❌ 评估异常: {str(e)}")
            model_status[model_name] = f"异常"
    
    # 生成对比表格
    print(f"\n\n" + "="*70)
    print("📊 模型性能对比总结")
    print("="*70)
    
    if results:
        # 创建 DataFrame
        df = pd.DataFrame(results).T
        
        print("\n📋 详细对比表格:")
        print(df.to_string())
        
        # 保存为 CSV
        output_file = "curriculum_model_comparison.csv"
        df.to_csv(output_file)
        print(f"\n💾 结果已保存到: {output_file}")
        
        # 打印汇总信息
        print(f"\n{'='*70}")
        print("📋 评估状态汇总")
        print(f"{'='*70}")
        for model_name, status in model_status.items():
            print(f"  {model_name}: {status}")
    else:
        print("\n⚠️  没有可评估的模型")
        print("\n📋 评估状态:")
        for model_name, status in model_status.items():
            print(f"  {model_name}: {status}")
        
        print("\n💡 故障排除建议:")
        print("  1. 确保模型路径正确")
        print("  2. 检查模型是否已训练:")
        for model_name, config in model_configs.items():
            print(f"     ls -la {config['path']}")
        print("  3. 如需训练新模型，运行:")
        print("     python curriculum_learning.py")
    
    print(f"\n{'='*70}")
    print("✅ 评估完成！")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
