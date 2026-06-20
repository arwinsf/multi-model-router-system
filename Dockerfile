# Official PyTorch image with CUDA 13.0 + cuDNN and CUDA toolkit (nvcc)
FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel

# Avoid interactive apt prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/hf-models
ENV TRANSFORMERS_CACHE=/workspace/hf-models
ENV HF_HUB_CACHE=/workspace/hf-models/hub
ENV WORKSPACE_ENV_FILE=/workspace/src/.env
ENV BASH_ENV=/etc/profile.d/workspace-env.sh

# Small set of convenient tools for "write python scripts" workflows
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    tmux \
    htop \
    ca-certificates \
    gcc \
    libc6-dev \
    libx11-6 \
    libxtst6 \
    xclip \
    xsel \
 && rm -rf /var/lib/apt/lists/*

# Workspace
WORKDIR /workspace

# Fuer bash-basierte Container-Kommandos (z.B. `bash -lc 'hf auth whoami'`)
# laden wir Projekt-Variablen automatisch aus dem gemounteten Workspace.
# Das Token selbst wird NICHT ins Image kopiert, sondern nur zur Laufzeit aus
# /workspace/src/.env uebernommen.
COPY docker/workspace-env.sh /etc/profile.d/workspace-env.sh

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --break-system-packages --upgrade pip \
 && python -m pip install --break-system-packages --no-cache-dir -r /tmp/requirements.txt

# BigCodeBench eval deps — strip exact version pins for Python 3.12 compat
RUN python -c "import urllib.request; open('/tmp/bcb-eval.txt','wb').write(urllib.request.urlopen('https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt').read())" \
 && sed -i 's/==.*//' /tmp/bcb-eval.txt \
 && python -m pip install --break-system-packages --no-cache-dir -r /tmp/bcb-eval.txt \
 || echo "WARN: some BigCodeBench eval deps could not be installed (non-fatal)"

# Keep the container alive so you can always docker exec into it
CMD ["bash", "-lc", "sleep infinity"]