import zipfile
import os

model_dir = "./bge-small-zh-v1.5"
zip_path = "./bge_model.zip"

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            full_path = os.path.join(root, file)
            # 保持目录结构，确保解压后出现 bge-small-zh-v1.5 文件夹
            arcname = os.path.relpath(full_path, start=os.path.dirname(model_dir))
            zf.write(full_path, arcname)

print(f"打包完成，大小：{os.path.getsize(zip_path) / 1024 / 1024:.2f} MB")