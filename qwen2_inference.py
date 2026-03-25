"""
Qwen2 KYC模型推理脚本

支持:
  - 单条推理
  - 批量推理
  - 交互式推理
  - REST API服务
"""

import os
import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import pandas as pd
from tqdm import tqdm


# ============================================================================
# 推理引擎
# ============================================================================

class QwenKYCInference:
    """Qwen2 KYC推理引擎"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """初始化推理引擎"""
        
        print(f"📦 加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        self.device = device
        self.model.eval()
        print("✅ 模型加载完成")
    
    def assess_kyc(
        self,
        kyc_text: str,
        max_length: int = 1024,
        top_p: float = 0.9,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """
        进行单条KYC评估
        
        Args:
            kyc_text: KYC材料文本
            max_length: 最大生成长度
            top_p: nucleus采样参数
            temperature: 温度参数
            do_sample: 是否使用采样
        
        Returns:
            风险评估结果
        """
        
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
                top_p=top_p,
                temperature=temperature,
                do_sample=do_sample,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取评估部分
        if "评估结果：" in response:
            assessment = response.split("评估结果：")[-1]
        else:
            assessment = response
        
        return assessment.strip()
    
    def batch_assess(
        self,
        kyc_texts: List[str],
        batch_size: int = 4,
        show_progress: bool = True,
    ) -> List[str]:
        """
        批量KYC评估
        
        Args:
            kyc_texts: KYC材料列表
            batch_size: 批大小
            show_progress: 是否显示进度
        
        Returns:
            评估结果列表
        """
        
        results = []
        
        if show_progress:
            iterator = tqdm(range(0, len(kyc_texts), batch_size), desc="Batch Assessment")
        else:
            iterator = range(0, len(kyc_texts), batch_size)
        
        for i in iterator:
            batch = kyc_texts[i:i+batch_size]
            
            for kyc_text in batch:
                assessment = self.assess_kyc(kyc_text)
                results.append(assessment)
        
        return results
    
    def extract_risk_decision(self, assessment: str) -> Dict:
        """
        从评估结果中提取风险决策
        
        Returns:
            包含风险等级、建议等的字典
        """
        
        decision = {
            'risk_level': None,
            'recommendation': None,
            'reasoning_steps': [],
        }
        
        # 提取风险等级
        if '低风险' in assessment or 'low risk' in assessment.lower():
            decision['risk_level'] = 'low'
        elif '高风险' in assessment or 'high risk' in assessment.lower():
            decision['risk_level'] = 'high'
        
        # 提取建议
        if '标准授信' in assessment:
            decision['recommendation'] = 'standard_credit'
        elif '严格控制' in assessment:
            decision['recommendation'] = 'strict_control'
        elif '拒绝' in assessment:
            decision['recommendation'] = 'reject'
        
        # 提取推理步骤
        steps = assessment.split('|')
        for step in steps:
            if '[' in step and ']' in step:
                decision['reasoning_steps'].append(step.strip())
        
        return decision


# ============================================================================
# 主要接口
# ============================================================================

def interactive_mode(engine: QwenKYCInference):
    """交互式推理模式"""
    
    print("\n" + "=" * 80)
    print("🎯 交互式KYC评估")
    print("=" * 80)
    print("输入KYC材料后，按两次回车提交\n")
    
    while True:
        print("输入KYC材料 (或 'quit' 退出):")
        lines = []
        
        while True:
            line = input()
            if line == 'quit':
                print("✅ 退出")
                return
            if line == '':
                if lines and lines[-1] == '':
                    lines.pop()
                    break
            lines.append(line)
        
        kyc_text = '\n'.join(lines)
        
        if not kyc_text.strip():
            continue
        
        print("\n⏳ 正在评估...")
        assessment = engine.assess_kyc(kyc_text)
        
        print("\n" + "-" * 60)
        print("【评估结果】")
        print(assessment)
        
        # 提取决策
        decision = engine.extract_risk_decision(assessment)
        print("\n【提取的决策】")
        print(f"  风险等级: {decision['risk_level']}")
        print(f"  建议: {decision['recommendation']}")
        print(f"  推理步骤数: {len(decision['reasoning_steps'])}")
        print("-" * 60 + "\n")


def file_mode(engine: QwenKYCInference, input_file: str, output_file: str):
    """文件模式：从CSV/JSONL读取，保存结果"""
    
    print(f"📥 加载数据: {input_file}")
    
    # 确定文件格式
    if input_file.endswith('.csv'):
        data = pd.read_csv(input_file)
        kyc_texts = data['kyc_complex_text'].tolist() if 'kyc_complex_text' in data.columns else data.iloc[:, 0].tolist()
    elif input_file.endswith('.jsonl'):
        kyc_texts = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                kyc_texts.append(record['prompt'])
    else:
        print("❌ 不支持的文件格式，请使用 .csv 或 .jsonl")
        return
    
    print(f"✅ 已加载 {len(kyc_texts)} 条记录")
    
    # 执行批量推理
    print("\n🔄 执行批量推理...")
    assessments = engine.batch_assess(kyc_texts)
    
    # 保存结果
    print(f"\n💾 保存结果: {output_file}")
    
    results = []
    for kyc, assessment in zip(kyc_texts, assessments):
        decision = engine.extract_risk_decision(assessment)
        results.append({
            'kyc_text': kyc[:100] + '...' if len(kyc) > 100 else kyc,
            'assessment': assessment,
            'risk_level': decision['risk_level'],
            'recommendation': decision['recommendation'],
        })
    
    if output_file.endswith('.json'):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"✅ 结果已保存")
    
    # 显示统计
    risk_counts = {}
    for r in results:
        risk_level = r['risk_level']
        risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
    
    print(f"\n【统计】")
    for level, count in risk_counts.items():
        print(f"  {level}: {count}")


def api_mode(engine: QwenKYCInference, port: int = 5000):
    """REST API模式"""
    
    app = Flask(__name__)
    
    @app.route('/assess', methods=['POST'])
    def assess():
        """KYC评估接口"""
        try:
            data = request.json
            kyc_text = data.get('kyc_text', '')
            
            if not kyc_text:
                return jsonify({'error': '缺少kyc_text'}), 400
            
            assessment = engine.assess_kyc(kyc_text)
            decision = engine.extract_risk_decision(assessment)
            
            return jsonify({
                'success': True,
                'assessment': assessment,
                'decision': decision,
            })
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
            }), 500
    
    @app.route('/batch', methods=['POST'])
    def batch():
        """批量评估接口"""
        try:
            data = request.json
            kyc_texts = data.get('kyc_texts', [])
            
            if not kyc_texts:
                return jsonify({'error': '缺少kyc_texts'}), 400
            
            assessments = engine.batch_assess(kyc_texts, show_progress=False)
            
            results = []
            for kyc, assessment in zip(kyc_texts, assessments):
                decision = engine.extract_risk_decision(assessment)
                results.append({
                    'assessment': assessment,
                    'decision': decision,
                })
            
            return jsonify({
                'success': True,
                'results': results,
            })
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
            }), 500
    
    @app.route('/health', methods=['GET'])
    def health():
        """健康检查"""
        return jsonify({'status': 'ok', 'model': 'Qwen2-7B-KYC'})
    
    print(f"\n🚀 启动API服务: http://localhost:{port}")
    print("\n【接口】")
    print(f"  POST /assess     - 单条评估")
    print(f"  POST /batch      - 批量评估")
    print(f"  GET  /health     - 健康检查")
    print("\n【示例】")
    print("  curl -X POST http://localhost:5000/assess \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"kyc_text\": \"...\"}'")
    
    app.run(host='0.0.0.0', port=port)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2 KYC推理")
    
    parser.add_argument('--model-path', default='./qwen2_kyc_grpo_model', help='模型路径')
    parser.add_argument('--mode', choices=['interactive', 'file', 'api'], default='interactive', help='运行模式')
    parser.add_argument('--input', help='输入文件 (file模式)')
    parser.add_argument('--output', default='kyc_assessment_results.json', help='输出文件')
    parser.add_argument('--port', type=int, default=5000, help='API端口')
    
    args = parser.parse_args()
    
    # 初始化引擎
    engine = QwenKYCInference(args.model_path)
    
    # 执行相应模式
    if args.mode == 'interactive':
        interactive_mode(engine)
    
    elif args.mode == 'file':
        if not args.input:
            print("❌ 文件模式需要 --input 参数")
            exit(1)
        file_mode(engine, args.input, args.output)
    
    elif args.mode == 'api':
        api_mode(engine, args.port)
