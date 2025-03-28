set -x
wget -qO- https://astral.sh/uv/install.sh | sh
uv venv pretrain-env --python 3.11 && source pretrain-env/bin/activate && uv pip install --upgrade pip
uv pip install vllm==0.8.2 --link-mode=copy
uv pip install -r pyproject.toml --link-mode=copy
uv pip install flash_attn --no-build-isolation --link-mode=copy
