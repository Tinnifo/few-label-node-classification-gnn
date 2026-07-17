# uv-based image. Mirrors the local `uv sync` setup so the container and a dev
# machine resolve identical dependencies from uv.lock.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# System deps occasionally needed to build torch-geometric extras from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install deps first (cached layer) from the lockfile, without the project or
# the notebook-only dev group. --frozen fails loudly if uv.lock is stale.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --no-dev

# torch>=2.6 defaults torch.load to weights_only=True, which rejects PyG's
# pickled Planetoid cache; restore the pre-2.6 behaviour.
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

COPY . .

# Default: the full sweep. Override at `docker run` for a single experiment, e.g.
#   docker run <img> uv run python src/train.py model=gcn method=vanilla dataset=cora
CMD ["uv", "run", "python", "src/train.py", "--multirun", "+experiment=full_grid"]
