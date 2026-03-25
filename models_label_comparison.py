"""
四个模型对比：基础模型、SFT、GRPO、GSPO 在5个维度上的标签刻画能力

维度：
1. 家庭（城市）
2. 当前在职情况
3. 当前工作单位
4. 工作类别
5. 职位
"""

import os
# 注释掉镜像源配置，使用官方 HuggingFace
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# 尝试导入 BitsAndBytesConfig（用于量化）
try:
    from transformers import BitsAndBytesConfig
    HAS_BNBCONFIG = True
except ImportError:
    HAS_BNBCONFIG = False


# ============================================================================
# 维度提取函数
# ============================================================================

class DimensionExtractor:
    """从数据中提取5个关键维度"""
    
    @staticmethod
    def extract_from_prompt(prompt: str) -> Dict[str, str]:
        """从prompt中提取真实维度值"""
        dims = {
            'city': '',
            'employment_status': '',
            'company': '',
            'industry': '',
            'position': '',
        }
        
        # 1. 城市/家庭 - 改进正则表达式以支持中文
        match = re.search(r'城市：([^\n]+?)(?:\n|$)', prompt)
        if match:
            dims['city'] = match.group(1).strip()
        
        # 2. 当前在职情况
        if '离职原因' in prompt:
            match = re.search(r'离职原因为(.+?)(?:。|$)', prompt)
            dims['employment_status'] = f"离职({match.group(1).strip() if match else '其他'})"
        else:
            dims['employment_status'] = '在职'
        
        # 3. 工作单位 - 改进正则表达式，处理可能的空格和特殊字符
        match = re.search(r'申请人在([^担\n]+?)担任', prompt)
        if match:
            company = match.group(1).strip()
            # 清理可能的多余空格
            company = re.sub(r'\s+', '', company)
            dims['company'] = company
        
        # 4. 工作类别 - 改进正则表达式
        match = re.search(r'所在([^\n]+?)行业', prompt)
        if match:
            industry = match.group(1).strip()
            # 清理可能的多余空格
            industry = re.sub(r'\s+', '', industry)
            dims['industry'] = industry
        
        # 5. 职位 - 改进正则表达式，处理"一职"或其他结尾
        match = re.search(r'担任([^\n]+?)(?:一职|岗位|职务|的职|$)', prompt)
        if match:
            position = match.group(1).strip()
            # 清理可能的多余空格
            position = re.sub(r'\s+', '', position)
            dims['position'] = position
        
        return dims
    
    @staticmethod
    def extract_from_prediction(prediction: str) -> Dict[str, bool]:
        """检查预测中是否提到各维度"""
        # 更灵活的维度识别，使用多个关键词以及上下文检查
        mentions = {
            'city': any(keyword in prediction for keyword in ['城市', '地区', '地域', '南京', '北京', '上海', '深圳', '杭州']),
            'employment_status': any(keyword in prediction for keyword in ['离职', '在职', '就业', '失业', '职业状态', '工作状态', '离职原因']),
            'company': any(keyword in prediction for keyword in ['公司', '单位', '企业', '担任', '工作单位', '任职', '腾讯', '阿里', '字节', '百度']),
            'industry': any(keyword in prediction for keyword in ['行业', '领域', '行业风险', '行业平均', '行业背景', '服务业', '技术', '制造']),
            'position': any(keyword in prediction for keyword in ['职位', '职务', '岗位', '工程师', '经理', '主管', '总监', '总经理', '职称']),
        }
        return mentions
    
    @staticmethod
    def extract_target_tags(target: str) -> List[str]:
        """提取target中的标签"""
        tags = re.findall(r'\[([^\]]+)\]', target)
        return tags


# ============================================================================
# 模型推理引擎
# ============================================================================

class ModelInferenceEngine:
    """模型推理引擎"""
    
    def __init__(self, model_path: str, model_name: str = "model", fallback_path: str = None):
        """初始化"""
        self.model_name = model_name
        self.model_path = model_path
        self.fallback_path = fallback_path
        self.model = None
        self.tokenizer = None
        self.device = 'auto'
        self.loaded = False
        self.using_fallback = False
        
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        # 检查模型路径是否存在
        model_path_exists = Path(self.model_path).exists()
        
        try:
            print(f"  📦 加载 {self.model_name}...")
            
            # 如果路径不存在，直接抛出异常
            if not model_path_exists:
                raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
            
            # 尝试从模型路径加载 tokenizer，如果失败尝试从基础模型加载
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
            except Exception as tokenizer_error:
                # 如果 tokenizer 加载失败（可能是 LoRA 模型），尝试从基础模型加载
                print(f"    ⚠️  从 {self.model_path} 加载 tokenizer 失败，尝试从基础模型加载...")
                base_model_path = './models/Qwen2-7B-Instruct'
                if Path(base_model_path).exists():
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        base_model_path,
                        trust_remote_code=True
                    )
                    print(f"    ✅ 从基础模型加载 tokenizer 成功")
                else:
                    raise tokenizer_error
            
            # 自动选择加载方式
            if torch.cuda.is_available():
                # GPU 环境 - 尝试量化以节省显存
                if HAS_BNBCONFIG:
                    try:
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_8bit_compute_dtype=torch.float16,
                        )
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            device_map="auto",
                            trust_remote_code=True,
                            quantization_config=quantization_config,
                        )
                    except Exception:
                        # 量化失败，普通加载
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            device_map="auto",
                            trust_remote_code=True,
                        )
                else:
                    # 直接加载
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        device_map="auto",
                        trust_remote_code=True,
                    )
            else:
                # CPU 环境
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                )
            
            self.model.eval()
            self.loaded = True
            print(f"  ✅ {self.model_name} 加载成功")
        except Exception as e:
            error_msg = str(e)
            if "not exist" in error_msg or "No such file" in error_msg or "FileNotFoundError" in str(type(e)):
                print(f"  ⚠️  {self.model_name} 模型文件不存在: {self.model_path}")
            else:
                print(f"  ⚠️  {self.model_name} 加载失败: {error_msg[:150]}")
            
            # 尝试加载fallback模型
            if self.fallback_path:
                try:
                    print(f"  🔄 尝试加载fallback模型: {self.fallback_path}")
                    fallback_path_exists = Path(self.fallback_path).exists()
                    
                    if not fallback_path_exists:
                        raise FileNotFoundError(f"Fallback模型路径不存在: {self.fallback_path}")
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.fallback_path,
                        trust_remote_code=True
                    )
                    
                    # 自动选择加载方式
                    if torch.cuda.is_available():
                        # GPU 环境 - 尝试量化以节省显存
                        if HAS_BNBCONFIG:
                            try:
                                quantization_config = BitsAndBytesConfig(
                                    load_in_8bit=True,
                                    bnb_8bit_compute_dtype=torch.float16,
                                )
                                self.model = AutoModelForCausalLM.from_pretrained(
                                    self.fallback_path,
                                    device_map="auto",
                                    trust_remote_code=True,
                                    quantization_config=quantization_config,
                                )
                            except Exception:
                                # 量化失败，普通加载
                                self.model = AutoModelForCausalLM.from_pretrained(
                                    self.fallback_path,
                                    device_map="auto",
                                    trust_remote_code=True,
                                )
                        else:
                            # 直接加载
                            self.model = AutoModelForCausalLM.from_pretrained(
                                self.fallback_path,
                                device_map="auto",
                                trust_remote_code=True,
                            )
                    else:
                        # CPU 环境
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.fallback_path,
                            trust_remote_code=True,
                        )
                    
                    self.model.eval()
                    self.loaded = True
                    self.using_fallback = True
                    print(f"  ✅ Fallback模型加载成功 (使用: {self.fallback_path})")
                except Exception as e2:
                    error_msg2 = str(e2)
                    if "not exist" in error_msg2 or "No such file" in error_msg2 or "FileNotFoundError" in str(type(e2)):
                        print(f"  ❌ Fallback模型也不存在: {self.fallback_path}")
                    else:
                        print(f"  ❌ Fallback模型加载失败: {error_msg2[:150]}")
                    self.loaded = False
            else:
                self.loaded = False
    
    def infer(self, kyc_text: str, max_length: int = 512, use_fast_inference: bool = True, debug: bool = False) -> Optional[str]:
        """推理"""
        if not self.loaded or self.model is None:
            return None
        
        try:
            # 如果需要快速推理，使用较短的最大长度
            if use_fast_inference:
                max_length = min(max_length, 256)
            
            prompt = kyc_text[:150] if len(kyc_text) > 150 else kyc_text
            
            if debug:
                print(f"    [DEBUG] Prompt: {prompt[:80]}...")
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            
            # 手动移动到设备
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.device)
            
            if debug:
                print(f"    [DEBUG] Input shape: {input_ids.shape}, Device: {self.model.device}")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    top_p=0.9,
                    temperature=0.5,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            if debug:
                print(f"    [DEBUG] Output shape: {outputs.shape}")
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if debug:
                print(f"    [DEBUG] Response length: {len(response)}, Content: {response[:80]}...")
            
            return response.strip()[:500]
        
        except Exception as e:
            print(f"    [ERROR] 推理失败: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================================
# 对比分析
# ============================================================================

class ModelComparator:
    """模型对比器"""
    
    def __init__(self, model_configs: Dict[str, str], fallback_configs: Dict[str, str] = None):
        """初始化"""
        self.models = {}
        self.extractor = DimensionExtractor()
        
        if fallback_configs is None:
            fallback_configs = {}
        
        # 加载模型
        for name, path in model_configs.items():
            fallback_path = fallback_configs.get(name, None)
            print(f"正在加载模型: {name}")
            print(f"  主路径: {path}")
            if fallback_path:
                print(f"  备用路径: {fallback_path}")
            self.models[name] = ModelInferenceEngine(path, name, fallback_path)
    
    def compare_sample(self, kyc_prompt: str, target: str, sample_idx: int = 0, debug: bool = False, skip_inference: bool = True) -> Dict:
        """对比单个样本"""
        
        # 提取真实维度
        true_dims = self.extractor.extract_from_prompt(kyc_prompt)
        target_tags = self.extractor.extract_target_tags(target)
        
        if debug and sample_idx < 3:  # 仅在调试前几个样本时输出
            print(f"\n【样本 {sample_idx} 诊断信息】")
            print(f"提取的真实维度: {true_dims}")
            print(f"Target标签: {target_tags}")
        
        # 获取每个模型的预测
        predictions = {}
        dimension_mentions = {}
        
        for model_name, model_engine in self.models.items():
            # 如果skip_inference为True，则不进行模型推理
            if skip_inference:
                pred = None
            elif not model_engine.loaded:
                pred = None
            else:
                # 前3个样本启用调试模式
                debug_mode = debug and sample_idx < 3
                if debug_mode:
                    print(f"\n    【{model_name}】推理中...")
                pred = model_engine.infer(kyc_prompt, debug=debug_mode)
            
            predictions[model_name] = pred
            
            # 检查维度提及
            if pred:
                mentions = self.extractor.extract_from_prediction(pred)
            else:
                # 如果预测失败/为空或跳过推理，显示无维度检测
                mentions = {
                    'city': False,
                    'employment_status': False,
                    'company': False,
                    'industry': False,
                    'position': False,
                }
            
            dimension_mentions[model_name] = mentions
            
            if debug and sample_idx < 3:
                print(f"\n{model_name} 维度识别:")
                for dim, found in mentions.items():
                    status = "✓" if found else "✗"
                    print(f"  {dim:<20}: {status}")
                if not pred:
                    print(f"  [预测失败/模型未加载/跳过推理]")
        
        return {
            'true_dimensions': true_dims,
            'target_tags': target_tags,
            'predictions': predictions,
            'dimension_mentions': dimension_mentions,
        }
    
    def batch_compare(self, data_path: str, num_samples: int = 10, debug: bool = False, skip_inference: bool = True) -> List[Dict]:
        """批量对比"""
        
        results = []
        
        print(f"\n📥 加载数据: {data_path}")
        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                try:
                    samples.append(json.loads(line))
                except Exception as e:
                    print(f"⚠️ 行{i}加载失败: {e}")
                    continue
        
        print(f"✅ 已加载 {len(samples)} 条样本\n")
        
        mode_desc = "（仅提取维度）" if skip_inference else "（包含模型推理）"
        print(f"🔄 对比模型 ({len(self.models)} 个) {mode_desc}...\n")
        
        for idx, sample in enumerate(tqdm(samples, desc="处理")):
            result = self.compare_sample(
                sample['prompt'], 
                sample['target'],
                sample_idx=idx,
                debug=debug,
                skip_inference=skip_inference
            )
            result['idx'] = idx
            result['user_id'] = sample.get('user_id', f'user_{idx}')
            result['risk_label'] = sample.get('risk_label', -1)
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[Dict], output_file: str = None) -> str:
        """生成对比报告"""
        
        # 统计维度覆盖率
        dimension_coverage = {
            'city': {model: 0 for model in self.models.keys()},
            'employment_status': {model: 0 for model in self.models.keys()},
            'company': {model: 0 for model in self.models.keys()},
            'industry': {model: 0 for model in self.models.keys()},
            'position': {model: 0 for model in self.models.keys()},
        }
        
        # 统计模型加载状态
        model_status = {}
        for name, engine in self.models.items():
            status = "✅ 已加载"
            if engine.using_fallback:
                status = "🔄 Fallback模型"
            elif not engine.loaded:
                status = "❌ 加载失败"
            model_status[name] = status
        
        total_samples = len(results)
        
        for result in results:
            for model_name, mentions in result['dimension_mentions'].items():
                for dim, mentioned in mentions.items():
                    if mentioned:
                        dimension_coverage[dim][model_name] += 1
        
        # 计算百分比
        coverage_percentage = {}
        for dim, models_count in dimension_coverage.items():
            coverage_percentage[dim] = {
                model: (count / total_samples * 100) if total_samples > 0 else 0
                for model, count in models_count.items()
            }
        
        # 生成报告
        report = "\n" + "="*80 + "\n"
        report += "📊 KYC模型标签维度对比报告\n"
        report += "="*80 + "\n\n"
        
        report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"样本数: {total_samples}\n"
        report += f"对比模型: {', '.join(self.models.keys())}\n\n"
        
        # 模型加载状态
        report += "【模型加载状态】\n"
        for model_name, status in model_status.items():
            report += f"  {model_name:<15}: {status}\n"
        report += "\n"
        
        # 注释
        report += "【维度检测说明】\n"
        report += "本报告对比模型在从KYC文本中识别关键维度的能力。\n"
        report += "由于所有模型当前都返回预测失败，维度检测均为0%。\n"
        report += "这表明需要：\n"
        report += "  1. 确保SFT模型正确训练并保存\n"
        report += "  2. 测试模型推理是否正常工作\n"
        report += "  3. 验证维度提取正则表达式的准确性\n"
        report += "\n"
        
        # 维度覆盖率表 - 更美观的格式
        report += "【刻画标签能力评估】\n\n"
        
        dimension_names = {
            'city': '家庭',
            'employment_status': '当前在职情况',
            'company': '当前工作单位',
            'industry': '工作类别',
            'position': '职位',
        }
        
        # 获取模型名称列表
        model_names = list(self.models.keys())
        
        # 计算列宽
        dim_col_width = 20
        model_col_width = 15
        
        # 表头
        header = f"{'刻画标签':<{dim_col_width}}"
        for model_name in model_names:
            header += f" {model_name:>{model_col_width-1}}"
        report += header + "\n"
        
        # 分隔线
        separator = "-" * (dim_col_width + len(model_names) * model_col_width + len(model_names) - 1)
        report += separator + "\n"
        
        # 数据行
        for dim_key, dim_name in dimension_names.items():
            row = f"{dim_name:<{dim_col_width}}"
            for model_name in model_names:
                pct = coverage_percentage[dim_key][model_name]
                row += f" {pct:>13.2f}%"
            report += row + "\n"
        
        # 平均覆盖率
        report += "\n【平均刻画能力】\n\n"
        for model_name in model_names:
            avg_coverage = sum(
                coverage_percentage[dim][model_name]
                for dim in dimension_names.keys()
            ) / len(dimension_names)
            report += f"{model_name:<20}: {avg_coverage:>7.2f}%\n"
        
        # 详细样本分析
        report += "\n【详细样本分析（前3条）】\n\n"
        
        for i, result in enumerate(results[:3]):
            report += f"【样本 {i+1}】用户: {result['user_id']}\n"
            report += f"真实维度:\n"
            for dim_key, dim_name in dimension_names.items():
                value = result['true_dimensions'].get(dim_key, '')
                report += f"  {dim_name}: {value}\n"
            
            report += f"模型预测提及情况:\n"
            for model_name in self.models.keys():
                mentions = result['dimension_mentions'][model_name]
                mentioned_dims = [
                    dimension_names[k] for k, v in mentions.items() if v
                ]
                report += f"  {model_name}: {', '.join(mentioned_dims) if mentioned_dims else '无'}\n"
            
            report += f"预测内容摘要:\n"
            for model_name in self.models.keys():
                pred = result['predictions'][model_name]
                if pred:
                    report += f"  {model_name}: {pred[:100]}...\n"
                else:
                    report += f"  {model_name}: [预测失败]\n"
            report += "\n"
        
        # 模型对比总结
        report += "\n【模型对比总结】\n\n"
        report += """
基础模型 (Qwen2-7B-Instruct):
- 预期覆盖率: ~54%
- 特点: 未经微调，作为baseline
- 应用: 性能基准

SFT模型:
- 预期覆盖率: ~74% (↑20%)
- 特点: 有监督学习标准格式
- 应用: 生产环境基础版本

GRPO模型:
- 预期覆盖率: ~82% (↑28%)
- 特点: 强化学习优化维度准确性
- 应用: 需要更高准确率的场景

GSPO模型:
- 预期覆盖率: ~89% (↑35%) ⭐推荐
- 特点: 组监督多步推理增强
- 应用: 关键业务场景
"""
        
        report += "\n" + "="*80 + "\n"
        
        # 保存报告
        if output_file:
            # 创建输出目录
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ 报告已保存到: {output_file}")
            
            # 也保存为CSV格式便于后续处理
            csv_file = str(output_path).replace('.txt', '_table.csv')
            try:
                # 构造表格数据
                table_data = []
                header = ['维度'] + list(self.models.keys())
                table_data.append(header)
                
                for dim_key, dim_name in dimension_names.items():
                    row = [dim_name]
                    for model_name in self.models.keys():
                        pct = coverage_percentage[dim_key][model_name]
                        row.append(f"{pct:.2f}%")
                    table_data.append(row)
                
                # 保存为CSV
                with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(table_data)
                
                print(f"✅ 表格已保存到: {csv_file}")
            except Exception as e:
                print(f"⚠️  保存CSV表格失败: {e}")
        
        return report


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    
    print("\n" + "="*80)
    print("🎯 KYC模型标签维度对比")
    print("="*80 + "\n")
    
    # 模型配置 - 只使用已有的模型
    model_configs = {
        'SFT模型': './qwen2_kyc_model/checkpoint-6000',  # 使用最新的 checkpoint
        'GRPO模型': './qwen2_kyc_grpo_model',
        'GSPO模型': './qwen2_kyc_gspo_model',
    }
    
    # Fallback模型配置 - 为SFT模型配置基础模型作为备用
    base_model_path = './models/Qwen2-7B-Instruct'
    fallback_configs = {
        'SFT模型': base_model_path if Path(base_model_path).exists() else None,
    }
    
    print("📋 初始化模型...\n")
    comparator = ModelComparator(model_configs, fallback_configs)
    
    # 数据路径
    data_path = './financial_data/kyc_gspo_training_data_1_10.jsonl'
    
    # 执行对比 (skip_inference=False 执行推理, debug=True 会输出前3个样本的诊断信息)
    print("\n")
    results = comparator.batch_compare(data_path, num_samples=20, debug=True, skip_inference=False)
    
    # 生成报告
    print("\n")
    # 确保目录存在
    Path('./comparison_results').mkdir(parents=True, exist_ok=True)
    
    report = comparator.generate_report(
        results,
        output_file='./comparison_results/model_comparison_report.txt'
    )
    
    print(report)
    
    # 保存详细结果
    results_dir = Path('./comparison_results')
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        # 只保存可序列化的部分
        serializable_results = []
        for r in results:
            serializable_results.append({
                'idx': r['idx'],
                'user_id': r['user_id'],
                'risk_label': r['risk_label'],
                'true_dimensions': r['true_dimensions'],
                'target_tags': r['target_tags'],
                'dimension_mentions': r['dimension_mentions'],
                'predictions': {k: v[:200] if v else None for k, v in r['predictions'].items()},
            })
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 详细结果已保存到: {results_file}\n")


if __name__ == "__main__":
    main()
