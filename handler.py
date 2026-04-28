#!/usr/bin/env python3
from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import boto3
import requests
import runpod
from PIL import Image, ImageDraw


COMFY_BASE_URL = os.environ.get("COMFY_BASE_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_DIR = Path(os.environ.get("COMFY_DIR", "/comfyui"))
COMFY_OUTPUT_DIR = Path(os.environ.get("COMFY_OUTPUT_DIR", "/comfyui/output"))
R2_PREFIX = os.environ.get("R2_PREFIX", "videos/serverless").strip("/")
WORKFLOW = COMFY_DIR / "custom_nodes/ComfyUI-LTXVideo/example_workflows/2.3/LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json"


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def request_json(path: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        COMFY_BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def wait_for_comfy(timeout_s: int = 900) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(COMFY_BASE_URL + "/", timeout=5) as resp:
                if 200 <= resp.status < 500:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError("ComfyUI API did not become reachable")


def ensure_input_image() -> None:
    path = COMFY_DIR / "input/example.png"
    path.parent.mkdir(exist_ok=True)
    if path.is_file():
        return
    img = Image.new("RGB", (512, 288), (28, 30, 36))
    d = ImageDraw.Draw(img)
    d.rectangle([32, 32, 480, 256], outline=(190, 180, 145), width=4)
    d.ellipse([205, 70, 307, 172], fill=(120, 170, 205))
    d.rectangle([230, 172, 282, 230], fill=(130, 95, 70))
    img.save(path)


def ancestors(nodes: dict[int, dict[str, Any]], links: dict[int, list[Any]], nid: int, out: set[int] | None = None) -> set[int]:
    out = out or set()
    if nid in out:
        return out
    out.add(nid)
    for inp in nodes[nid].get("inputs", []):
        lid = inp.get("link")
        if lid in links:
            ancestors(nodes, links, links[lid][1], out)
    return out


def ordered_inputs(info: dict[str, Any]) -> list[str]:
    order: list[str] = []
    inp_order = info.get("input_order") or {}
    for section in ("required", "optional"):
        order += inp_order.get(section) or []
    if not order:
        inputs = info.get("input") or {}
        for section in ("required", "optional"):
            order += list((inputs.get(section) or {}).keys())
    return order


def build_ltx_test_prompt(job_input: dict[str, Any]) -> dict[str, Any]:
    ensure_input_image()
    workflow = json.loads(WORKFLOW.read_text())
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    keep = ancestors(nodes, links, 4852)
    object_info = request_json("/object_info")

    filename_prefix = job_input.get("filename_prefix") or "ltx_serverless_test"
    prompt_text = job_input.get("prompt_text") or (
        "A small chrome rocket toy lifts off from a wooden desk, warm cinematic light, "
        "playful smoke puffs, smooth camera move."
    )
    negative = job_input.get("negative_prompt") or "ugly, blurry, low quality, distorted, jittery"
    width = int(job_input.get("width") or 384)
    height = int(job_input.get("height") or 224)
    frames = int(job_input.get("frames") or 25)
    seed = int(job_input.get("seed") or 1024)

    patches = {
        3059: [width, height, frames, 1],
        4979: [frames, "fixed"],
        3940: ["LTX-Video/ltx-2.3-22b-distilled-1.1.safetensors"],
        4010: ["LTX-Video/ltx-2.3-22b-distilled-1.1.safetensors"],
        4960: [
            "gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors",
            "LTX-Video/ltx-2.3-22b-distilled-1.1.safetensors",
            "default",
        ],
        2483: [prompt_text],
        2612: [negative],
        4832: [seed, "fixed"],
        4852: [filename_prefix, "auto", "auto"],
    }
    for nid, vals in patches.items():
        if nid in nodes:
            nodes[nid]["widgets_values"] = vals

    prompt: dict[str, Any] = {}
    for nid in sorted(keep):
        node = nodes[nid]
        linked = {}
        widget_linked = set()
        for inp in node.get("inputs", []):
            name = inp.get("name")
            lid = inp.get("link")
            if lid in links:
                link = links[lid]
                linked[name] = [str(link[1]), link[2]]
                if inp.get("widget"):
                    widget_linked.add(name)

        values = list(node.get("widgets_values") or [])
        value_index = 0
        inputs = dict(linked)
        for name in ordered_inputs(object_info[node["type"]]):
            if name in linked:
                if name in widget_linked and value_index < len(values):
                    value_index += 1
                continue
            if value_index < len(values):
                inputs[name] = values[value_index]
                value_index += 1
        prompt[str(nid)] = {"class_type": node["type"], "inputs": inputs}

    prompt["4960"] = {
        "class_type": "LTXVGemmaCLIPModelLoader",
        "inputs": {
            "gemma_path": "gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors",
            "ltxv_path": "LTX-Video/ltx-2.3-22b-distilled-1.1.safetensors",
            "max_length": 1024,
        },
    }
    prompt["4828"]["inputs"]["model"] = ["3940", 0]
    prompt.pop("4922", None)
    if "4981" in prompt:
        prompt["4981"]["inputs"]["resize_type"] = "scale longer dimension"
        prompt["4981"]["inputs"]["resize_type.longer_size"] = 512
        prompt["4981"]["inputs"]["scale_method"] = "lanczos"
    return prompt


def submit_prompt(prompt: dict[str, Any]) -> str:
    payload = {"prompt": prompt, "client_id": "runpod-ltx-r2-" + uuid.uuid4().hex}
    response = request_json("/prompt", payload)
    return response["prompt_id"]


def wait_for_history(prompt_id: str, timeout_s: int) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        history = request_json(f"/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(2)
    raise TimeoutError(f"ComfyUI prompt did not finish within {timeout_s}s: {prompt_id}")


def r2_client():
    session_token = os.environ.get("R2_SESSION_TOKEN")
    kwargs = {}
    if session_token:
        kwargs["aws_session_token"] = session_token
    return boto3.client(
        "s3",
        endpoint_url=f"https://{require_env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=require_env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=require_env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        **kwargs,
    )


def output_items(history: dict[str, Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for node_output in (history.get("outputs") or {}).values():
        for value in node_output.values():
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, dict) and item.get("filename"):
                    items.append({
                        "filename": item["filename"],
                        "subfolder": item.get("subfolder") or "",
                        "type": item.get("type") or "output",
                    })
    dedup = {(i["filename"], i["subfolder"], i["type"]): i for i in items}
    return list(dedup.values())


def upload_file(path: Path, key: str) -> dict[str, Any]:
    worker_url = os.environ.get("R2_WORKER_UPLOAD_URL", "").rstrip("/")
    worker_token = os.environ.get("R2_UPLOAD_TOKEN")
    if worker_url and worker_token:
        return upload_file_via_worker(path, key, worker_url, worker_token)

    bucket = require_env("R2_BUCKET")
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    extra_args = {
        "ContentType": content_type,
        "CacheControl": os.environ.get("R2_CACHE_CONTROL", "public, max-age=31536000, immutable"),
        "StorageClass": os.environ.get("R2_STORAGE_CLASS", "STANDARD"),
    }
    r2_client().upload_file(str(path), bucket, key, ExtraArgs=extra_args)
    public_base_url = os.environ.get("R2_PUBLIC_BASE_URL", "").rstrip("/")
    return {
        "filename": path.name,
        "bucket": bucket,
        "key": key,
        "uri": f"r2://{bucket}/{key}",
        "public_url": f"{public_base_url}/{key}" if public_base_url else None,
        "bytes": path.stat().st_size,
    }


def upload_file_via_worker(path: Path, key: str, worker_url: str, token: str) -> dict[str, Any]:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded_key = "/".join(urllib.parse.quote(part, safe="") for part in key.split("/"))
    with path.open("rb") as fh:
        resp = requests.put(
            f"{worker_url}/upload/{encoded_key}",
            data=fh,
            headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": content_type,
            "Cache-Control": os.environ.get("R2_CACHE_CONTROL", "public, max-age=31536000, immutable"),
            },
            timeout=300,
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"R2 worker upload failed: {resp.status_code} {resp.text[:1000]}")
    result = resp.json()
    public_base_url = os.environ.get("R2_PUBLIC_BASE_URL", "").rstrip("/")
    return {
        "filename": path.name,
        "bucket": result.get("bucket", "ltx-video-outputs"),
        "key": result.get("key", key),
        "uri": result.get("uri", f"r2://ltx-video-outputs/{key}"),
        "public_url": f"{public_base_url}/{key}" if public_base_url else None,
        "bytes": path.stat().st_size,
    }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input") or {}
    wait_for_comfy(int(os.environ.get("COMFY_START_TIMEOUT_S", "900")))

    prompt = job_input.get("prompt")
    if not isinstance(prompt, dict):
        prompt = build_ltx_test_prompt(job_input)

    timeout_s = int(job_input.get("timeout_s") or os.environ.get("COMFY_TIMEOUT_S", "1800"))
    prompt_id = submit_prompt(prompt)
    history = wait_for_history(prompt_id, timeout_s)

    uploads = []
    upload_errors = []
    for item in output_items(history):
        if item["type"] == "temp":
            continue
        path = COMFY_OUTPUT_DIR / item["subfolder"] / item["filename"]
        if not path.is_file():
            continue
        key = f"{R2_PREFIX}/{prompt_id}/{item['filename']}"
        try:
            uploads.append(upload_file(path, key))
        except Exception as exc:
            upload_errors.append({
                "filename": item["filename"],
                "key": key,
                "error": str(exc),
                "local_path": str(path),
                "bytes": path.stat().st_size,
            })

    result = {
        "prompt_id": prompt_id,
        "status": (history.get("status") or {}).get("status_str"),
        "outputs": uploads,
    }
    if upload_errors:
        result["upload_errors"] = upload_errors
    return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
