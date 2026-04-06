import argparse
import os
from pathlib import Path

import torch
from diffusers import CogVideoXPipeline
from transformers import BitsAndBytesConfig
import imageio


def resolve_device() -> str:
    """Prefer xpu, then cuda, then cpu for broader compatibility."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_dtype(device: str) -> torch.dtype:
    """Use float32 on CPU and XPU; bfloat16 only on CUDA."""
    if device in ("cpu", "xpu"):
        return torch.float32
    return torch.bfloat16


def default_cache_dir() -> Path:
    if os.name == "nt":
        return Path(r"G:\HuggingFace\Model")
    return Path("./model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CogVideoX text-to-video runner")
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir(), help="Model cache directory")
    parser.add_argument("--quantized", dest="quantized", action="store_true", help="Enable 4-bit quantized load")
    parser.add_argument("--no-quantized", dest="quantized", action="store_false", help="Disable 4-bit quantized load")
    parser.set_defaults(quantized=True)
    parser.add_argument("--load-retries", type=int, default=2, help="Retry count for model loading failures")
    return parser.parse_args()


def prepare_cache(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Keep huggingface cache under a known location on Windows/Linux/macOS.
    os.environ["HF_HOME"] = str(cache_dir)
    return cache_dir


def load_pipeline(
    model_id: str,
    dtype: torch.dtype,
    cache_dir: Path,
    quantized: bool,
    load_retries: int,
) -> CogVideoXPipeline:
    base_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "cache_dir": str(cache_dir),
    }

    last_error: Exception | None = None
    attempts = max(1, load_retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            if quantized:
                return CogVideoXPipeline.from_pretrained(
                    model_id,
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=dtype,
                        bnb_4bit_quant_type="nf4",
                    ),
                    **base_kwargs,
                )
            return CogVideoXPipeline.from_pretrained(model_id, **base_kwargs)
        except ValueError as exc:
            if quantized and "PipelineQuantizationConfig" in str(exc):
                print("Quantization config is incompatible with this diffusers version, fallback to non-quantized load.")
                quantized = False
                last_error = exc
                continue
            last_error = exc
        except Exception as exc:  # noqa: BLE001
            last_error = exc

        if attempt < attempts:
            print(f"Model load failed (attempt {attempt}/{attempts}), retrying: {last_error}")

    raise RuntimeError(f"Failed to load model after {attempts} attempts: {last_error}")


def move_pipeline_to_device(pipe: CogVideoXPipeline, target_device: str) -> tuple[CogVideoXPipeline, str]:
    """Move pipeline to target device and gracefully fallback on OOM."""
    try:
        pipe = pipe.to(target_device)
        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae = pipe.vae.to(target_device)
        return pipe, target_device
    except torch.OutOfMemoryError:
        if target_device != "cpu":
            print(f"{target_device} out of memory, fallback to cpu.")
            torch.xpu.empty_cache() if hasattr(torch, "xpu") else None
            pipe = pipe.to("cpu")
            if hasattr(pipe, "vae") and pipe.vae is not None:
                pipe.vae = pipe.vae.to("cpu")
            return pipe, "cpu"
        raise


args = parse_args()
device = resolve_device()
dtype = resolve_dtype(device)
cache_dir = prepare_cache(args.cache_dir)

# ======================================
# 加载模型（原生无封装）
# ======================================
model_id = "THUDM/CogVideoX-2B"
pipe = load_pipeline(
    model_id=model_id,
    dtype=dtype,
    cache_dir=cache_dir,
    quantized=args.quantized,
    load_retries=args.load_retries,
)

# ======================================
# ✅ 新版：直接用 torch 原生 xpu（无ipex）
# ======================================
pipe, device = move_pipeline_to_device(pipe, device)
print(f"Using device: {device}")
print(f"Using cache dir: {cache_dir}")

# 显存优化（可选，不是所有 pipeline 都支持）
if hasattr(pipe, "enable_vae_slicing"):
    pipe.enable_vae_slicing()
if hasattr(pipe, "enable_attention_slicing"):
    pipe.enable_attention_slicing(slice_size=1)
if device in ("xpu", "cuda") and hasattr(pipe, "enable_model_cpu_offload"):
    # Offload can significantly reduce peak VRAM usage on small GPUs.
    pipe.enable_model_cpu_offload()

# ======================================
# 提示词
# ======================================
prompt = "一个女孩在樱花树下散步，微风拂动花瓣，电影质感，4K，细腻光影，流畅运动"
negative_prompt = "模糊，低分辨率，失真，变形，模糊运动，水印，文字"

# ======================================
# 生成视频
# ======================================
video_frames = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=4,
    width=192,
    height=112,
    num_inference_steps=4,
    guidance_scale=4.5,
).frames[0]

# 保存
imageio.mimsave("cogvideo_hq.mp4", video_frames, fps=6)
print("✅ 视频已生成：cogvideo_hq.mp4")