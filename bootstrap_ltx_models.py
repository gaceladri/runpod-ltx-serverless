#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


VOLUME = Path(os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume"))
MODELS = VOLUME / "models"
LTX_CKPT = MODELS / "checkpoints/LTX-Video/ltx-2.3-22b-distilled-1.1.safetensors"
GEMMA_DIR = MODELS / "text_encoders/gemma-3-12b-it-qat-q4_0-unquantized"
COMFY_DIR = Path(os.environ.get("COMFY_DIR", "/comfyui"))


def require_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required to populate the network volume")
    return token


def ensure_volume() -> None:
    if not VOLUME.is_dir():
        raise SystemExit(f"network volume is not mounted at {VOLUME}")
    for rel in (
        "checkpoints/LTX-Video",
        "text_encoders",
        "vae",
        "clip",
        "loras",
        "configs",
    ):
        (MODELS / rel).mkdir(parents=True, exist_ok=True)


def patch_comfy_paths() -> None:
    p = COMFY_DIR / "extra_model_paths.yaml"
    if not p.is_file():
        return
    s = p.read_text()
    if "text_encoders:" not in s:
        p.write_text(s.rstrip() + "\n  text_encoders: models/text_encoders/\n")


def patch_ltx_q4_config_case() -> None:
    p = COMFY_DIR / "custom_nodes/ComfyUI-LTXVideo/text_embeddings_connectors.py"
    if not p.is_file():
        return
    s = p.read_text()
    needle = '            actual = transformer_config.get(key)\n'
    insert = (
        '            actual = transformer_config.get(key)\n'
        '            if key == "text_encoder_norm_type" and isinstance(actual, str):\n'
        '                actual = actual.lower()\n'
    )
    if "actual = actual.lower()" not in s:
        p.write_text(s.replace(needle, insert, 1))


def download_models() -> None:
    token = require_token()

    if not LTX_CKPT.is_file():
        print(f"ltx-bootstrap: downloading {LTX_CKPT.name} to network volume")
        hf_hub_download(
            repo_id="Lightricks/LTX-2.3",
            filename="ltx-2.3-22b-distilled-1.1.safetensors",
            local_dir=str(LTX_CKPT.parent),
            token=token,
        )
    else:
        print(f"ltx-bootstrap: found {LTX_CKPT}")

    marker = GEMMA_DIR / "model-00001-of-00005.safetensors"
    if not marker.is_file():
        print(f"ltx-bootstrap: downloading Gemma QAT/Q4 to {GEMMA_DIR}")
        snapshot_download(
            repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
            local_dir=str(GEMMA_DIR),
            token=token,
        )
    else:
        print(f"ltx-bootstrap: found {GEMMA_DIR}")


def main() -> int:
    ensure_volume()
    patch_comfy_paths()
    patch_ltx_q4_config_case()
    download_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
