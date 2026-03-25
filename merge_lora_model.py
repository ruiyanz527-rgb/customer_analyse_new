#!/usr/bin/env python
"""
合并 LoRA 适配器和基础模型，生成完整的 SFT 模型文件
"""

import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import shutil

def merge_lora_model(
    base_model_path: str,
    lora_model_path: str,
    output_path: str,
    push_to_hub: bool = False,
):
    """
    合并 LoRA 模型和基础模型
    
    Args:
        base_model_path: 基础模型路径
        lora_model_path: LoRA 模型路径
        output_path: 输出路径
        push_to_hub: 是否推送到 HuggingFace Hub
    """
    
    print("\n" + "="*80)
    print("🔄 合并 LoRA 模型和基础模型")
    print("="*80 + "\n")
    
    # 检查路径
    if not Path(base_model_path).exists():
        raise FileNotFoundError(f"基础模型不存在: {base_model_path}")
    if not Path(lora_model_path).exists():
        raise FileNotFoundError(f"LoRA 模型不存在: {lora_model_path}")
    
    print(f"📂 基础模型: {base_model_path}")
    print(f"📂 LoRA 模型: {lora_model_path}")
    print(f"📂 输出路径: {output_path}\n")
    
    # 创建输出目录
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # 1. 加载基础模型
    print("📦 加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("✅ 基础模型加载成功")
    
    # 2. 加载 LoRA 适配器
    print("\n📌 加载 LoRA 适配器...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        device_map="auto",
    )
    print("✅ LoRA 适配器加载成功")
    
    # 3. 合并模型
    print("\n🔀 合并模型权重...")
    merged_model = model.merge_and_unload()
    print("✅ 模型合并成功")
    
    # 4. 保存合并后的模型
    print("\n💾 保存合并后的模型...")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    print("✅ 模型已保存")
    
    # 5. 保存 tokenizer 和配置文件
    print("\n📋 保存 tokenizer 和配置文件...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(output_path)
    print("✅ Tokenizer 和配置已保存")
    
    # 6. 复制其他必要文件
    print("\n📄 复制其他配置文件...")
    for file in ['chat_template.jinja', 'generation_config.json']:
        src = Path(lora_model_path) / file
        dst = Path(output_path) / file
        if src.exists():
            shutil.copy(src, dst)
            print(f"  ✅ 复制 {file}")
    
    print("\n" + "="*80)
    print("✅ 合并完成！")
    print("="*80)
    print(f"\n📍 合并后的模型已保存到: {output_path}")
    print(f"\n你现在可以直接使用这个路径加载模型:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_path}')")
    print("\n")


if __name__ == '__main__':
    # 配置
    base_model_path = './models/Qwen2-7B-Instruct'
    lora_model_path = './qwen2_kyc_model_merged_full'
    output_path = './qwen2_kyc_model_merged_full'
    
    try:
        merge_lora_model(
            base_model_path=base_model_path,
            lora_model_path=lora_model_path,
            output_path=output_path,
        )
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
