# download_model.py
import os
from pathlib import Path

# 使用国内镜像站加速下载（如果不需要可以注释掉下一行）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer


def download_model():
    model_name = "BAAI/bge-small-zh-v1.5"
    save_path = Path("./bge-small-zh-v1.5")

    print(f"开始下载模型: {model_name}")
    model = SentenceTransformer(model_name)

    # 保存模型到指定目录
    model.save_pretrained(save_path)

    # 计算并打印模型文件夹大小
    total_size = sum(f.stat().st_size for f in save_path.rglob('*') if f.is_file())
    print(f"模型下载完成！保存路径: {save_path.resolve()}")
    print(f"总大小: {total_size / 1024 / 1024:.2f} MB")

    # 可选：验证关键文件存在
    required_files = ["model.safetensors", "config.json", "tokenizer.json"]
    missing = [f for f in required_files if not (save_path / f).exists()]
    if missing:
        print(f"警告：缺少以下文件: {missing}")
    else:
        print("所有必需文件均已下载。")


if __name__ == "__main__":
    download_model()