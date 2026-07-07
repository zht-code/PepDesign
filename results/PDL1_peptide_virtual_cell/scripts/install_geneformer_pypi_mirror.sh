#!/usr/bin/env bash
# 在无法访问 huggingface.co 时，用 hf-mirror 拉取 Geneformer 源码并本地安装（不依赖 git clone 官方 URL）。
# 依赖: pip install huggingface_hub
# 注意：若用「sh 本脚本」运行，bash 专有选项必须省略（dash 不支持 pipefail）。
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${GENEFORMER_SRC_DIR:-$ROOT/vendor/Geneformer}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "将下载源码到: $SRC"

python3 << PY
from pathlib import Path
from huggingface_hub import snapshot_download

dst = Path("$SRC").resolve()
dst.parent.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id="ctheodoris/Geneformer",
    local_dir=str(dst),
    local_dir_use_symlinks=False,
)
print("snapshot OK:", dst)
PY

echo "安装 Geneformer 包（可编辑模式便于调试）..."
pip install -e "$SRC"
# 与官方 vendor/Geneformer/requirements.txt 一致；过新的 transformers 会缺 SpecialTokensMixin
pip install "transformers==4.46" --upgrade
echo "完成。验证: python3 -c \"import geneformer; print(geneformer.__file__)\""
