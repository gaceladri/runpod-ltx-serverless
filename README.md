# LTX Serverless Image

Custom RunPod Serverless image for:

- ComfyUI worker base image `runpod/worker-comfyui:5.7.1-base`
- `ComfyUI-LTXVideo` pinned at `2acf7af8991f33b5cc06ec26753cb6e88e057d04`
- LTX-2.3 distilled checkpoint and Gemma QAT/Q4 text encoder on a RunPod network volume
- Output upload to Cloudflare R2 via S3-compatible credentials

The network volume is expected at `/runpod-volume`. On first boot the image downloads:

- `/runpod-volume/models/checkpoints/LTX-Video/ltx-2.3-22b-distilled-1.1.safetensors`
- `/runpod-volume/models/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized/`

Required endpoint environment:

```bash
HF_TOKEN
R2_ACCOUNT_ID
R2_ACCESS_KEY_ID
R2_SECRET_ACCESS_KEY
R2_BUCKET=ltx-video-outputs
```

Build target image:

```text
ghcr.io/gaceladri/runpod-ltx-serverless:latest
```

