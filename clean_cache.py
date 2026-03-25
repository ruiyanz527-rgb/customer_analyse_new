import torch
print("清空 CUDA 缓存...")
torch.cuda.empty_cache()
print("✅ 完成")