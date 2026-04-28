FROM runpod/worker-comfyui:5.7.1-base

ENV PYTHONUNBUFFERED=1
ENV PIP_PREFER_BINARY=1

WORKDIR /comfyui

RUN git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git custom_nodes/ComfyUI-LTXVideo \
    && cd custom_nodes/ComfyUI-LTXVideo \
    && git fetch --depth 1 origin 2acf7af8991f33b5cc06ec26753cb6e88e057d04 \
    && git checkout 2acf7af8991f33b5cc06ec26753cb6e88e057d04 \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir boto3 huggingface_hub \
    && python - <<'PY'
from pathlib import Path
p = Path("/comfyui/custom_nodes/ComfyUI-LTXVideo/text_embeddings_connectors.py")
s = p.read_text()
needle = '            actual = transformer_config.get(key)\n'
insert = (
    '            actual = transformer_config.get(key)\n'
    '            if key == "text_encoder_norm_type" and isinstance(actual, str):\n'
    '                actual = actual.lower()\n'
)
if "actual = actual.lower()" not in s:
    p.write_text(s.replace(needle, insert, 1))
PY

RUN python - <<'PY'
from pathlib import Path
p = Path("/comfyui/extra_model_paths.yaml")
s = p.read_text()
if "text_encoders:" not in s:
    s = s.rstrip() + "\n  text_encoders: models/text_encoders/\n"
    p.write_text(s)
PY

COPY bootstrap_ltx_models.py /usr/local/bin/bootstrap_ltx_models.py
COPY handler.py /handler.py
COPY start_ltx_serverless.sh /usr/local/bin/start_ltx_serverless.sh

RUN chmod +x /usr/local/bin/bootstrap_ltx_models.py /usr/local/bin/start_ltx_serverless.sh

CMD ["/usr/local/bin/start_ltx_serverless.sh"]

