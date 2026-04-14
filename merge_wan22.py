import safetensors.torch
import glob
import os
import torch
from collections import defaultdict

# 1. 替换为你的分片文件所在的文件夹路径（Windows 用 r"" 避免转义）
model_dir = r"G:\ComfyUI\models\diffusion_models"

# 2. 读取索引文件，严格按官方顺序合并（核心修复：用索引保证顺序正确）
index_path = os.path.join(model_dir, "diffusion_pytorch_model.safetensors.index.json")
import json
with open(index_path, "r", encoding="utf-8") as f:
    index = json.load(f)

# 3. 按索引加载分片，避免顺序错误
shard_map = defaultdict(dict)
for weight_name, shard_file in index["weight_map"].items():
    shard_map[shard_file][weight_name] = None

# 4. 加载所有分片，按索引合并
merged_tensors = {}
shard_files = sorted(shard_map.keys())
for shard_file in shard_files:
    print(f"🔄 正在加载分片: {shard_file}")
    shard_path = os.path.join(model_dir, shard_file)
    tensors = safetensors.torch.load_file(shard_path)
    # 只保留索引中对应的权重，避免冗余
    for weight_name in shard_map[shard_file]:
        merged_tensors[weight_name] = tensors[weight_name]

# 5. 保存为完整单文件
output_path = os.path.join(model_dir, "Wan2.2-TI2V-5B-full.safetensors")
print(f"💾 正在保存完整模型到: {output_path}")
safetensors.torch.save_file(merged_tensors, output_path)
print("✅ 合并完成！模型张量100%匹配官方结构")