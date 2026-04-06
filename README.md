# HuggingFacePlay

Simple personal play ai model with hugging face platform

The model save directory are different for MacOS and Windows.

The MacOS directory is: `./model/**`

The Windows is directory: `G:\HuggingFace\Model\**`

## Installation

Choose the appropriate PyTorch version for your hardware:

### Intel Arc GPU (XPU)

```bash
pip install -r requirements-xpu.txt
```

### NVIDIA GPU (CUDA 12.1)

```bash
pip install -r requirements-cuda.txt
```

### CPU only

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

Run with default settings (quantized on, Windows cache defaults to `G:\HuggingFace\Model`):

```bash
python test.py
```

Disable quantization (recommended on incompatible diffusers/bitsandbytes combinations):

```bash
python test.py --no-quantized
```

Use a custom model cache directory:

```bash
python test.py --cache-dir D:\ModelCache --no-quantized --load-retries 3
```


