FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# CPU-only PyTorch keeps the image small (~1 GB instead of ~5 GB with CUDA)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Non-root user; mount ./runs to keep training outputs
RUN useradd -m ppo && mkdir -p /app/runs && chown -R ppo /app
USER ppo

# Default: show the training CLI. Override the command to train or serve, e.g.
#   docker run --rm -v $PWD/runs:/app/runs sim2real \
#       python -m robust_rl.train --env-id InvertedPendulum-v5 --total-timesteps 40960 --log-dir runs/docker
#   docker run -p 8000:8000 -v $PWD/runs/smoke:/models -e PPO_MODEL_PATH=/models/model.pt sim2real \
#       python -m uvicorn --factory robust_rl.serve:create_app --host 0.0.0.0 --port 8000
CMD ["python", "-m", "robust_rl.train", "--help"]
