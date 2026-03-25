import json
from pathlib import Path
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("🔀 合并 LoRA adapter 到基座模型...")
print()

sft_adapter_path = Path("./qwen2_kyc_model")
base_model_path = Path("./models/Qwen2-7B-Instruct")
output_path = Path("./qwen2_kyc_model_merged")

# 检查是否是 LoRA adapter
adapter_config = sft_adapter_path / "adapter_config.json"
if not adapter_config.exists():
    print("⚠️  不是 LoRA adapter 格式，尝试直接使用")
    print("如果还是出错，请检查 SFT 训练是否正确保存模型")
    exit(1)

print(f"基座模型: {base_model_path}")
print(f"LoRA adapter: {sft_adapter_path}")
print(f"输出路径: {output_path}")
print()

try:
    print("📥 加载基座模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)
    print("✓ 基座模型加载完成")
    
    print("📥 加载 LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base_model, str(sft_adapter_path))
    print("✓ LoRA adapter 加载完成")
    
    print("🔀 合并 LoRA 权重到基座模型...")
    merged_model = peft_model.merge_and_unload()
    print("✓ 合并完成")
    
    print(f"💾 保存完整模型到: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print("✓ 模型保存完成")
    
    # 清理内存
    del base_model, peft_model, merged_model
    torch.cuda.empty_cache()
    
    print()
    print("✅ 合并完成！")
    print()
    print("下一步：")
    print(f"1. 更新 GRPO 配置中的模型路径为: {output_path}")
    print("2. 重新运行 GRPO 训练: python3 qwen2_grpo_trainer.py")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
