import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen2-7B-Instruct"
print(f"Downloading {model_id}...")
snapshot_download(
    repo_id=model_id,
    local_dir=f"./models/{model_id.split('/')[-1]}",
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=4
)
print("Download completed!")
