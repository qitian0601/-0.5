#!/bin/bash
#SBATCH --job-name=download-qwen3vl30b-aria2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/download-qwen3vl30b-%j.out
#SBATCH --error=outputs/slurm/download-qwen3vl30b-%j.err

set -Eeuo pipefail

REPO_ROOT="/public/home/chenglongyan/workspace/lerobot"
PYTHON="/public/home/chenglongyan/apps/miniconda3/envs/lerobot/bin/python"
ARIA2C="${REPO_ROOT}/.tools/aria2-1.37.0/bin/aria2c"
MODEL_ID="Qwen/Qwen3-VL-30B-A3B-Instruct"
MODEL_REVISION="9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"
MODELS_ROOT="${REPO_ROOT}/hf_downloads/models/Qwen3-VL-30B-A3B-Instruct"
JOB_CACHE_ROOT="${TMPDIR:-/tmp}/qwen3vl_download_${SLURM_JOB_ID}"
MODEL_BASE_URL="https://hf-mirror.com/${MODEL_ID}/resolve/${MODEL_REVISION}"

cd "$REPO_ROOT"
mkdir -p "$MODELS_ROOT" "$JOB_CACHE_ROOT" outputs/slurm
export HF_HOME="$JOB_CACHE_ROOT/hf_home"
export HF_HUB_CACHE="$JOB_CACHE_ROOT/hub"
export XDG_CACHE_HOME="$JOB_CACHE_ROOT/xdg_cache"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$XDG_CACHE_HOME"

test -x "$ARIA2C"
"$ARIA2C" --version | head -1

"$PYTHON" - <<'PY'
import huggingface_hub
import transformers
from transformers import Qwen3VLMoeForConditionalGeneration

print(f"huggingface_hub={huggingface_hub.__version__}")
print(f"transformers={transformers.__version__}")
print(f"model_class={Qwen3VLMoeForConditionalGeneration.__name__}")
PY

TARGET="$MODELS_ROOT/$MODEL_REVISION"
PARTIAL="$MODELS_ROOT/.${MODEL_REVISION}.aria2.partial"
INPUT_FILE="$PARTIAL/aria2-input.txt"

if [[ -d "$TARGET" ]]; then
  test -f "$TARGET/config.json"
  find "$TARGET" -type f -name '*.safetensors' -print -quit | grep -q .
  echo "Model revision already available: $TARGET"
  exit 0
fi

mkdir -p "$PARTIAL"
MODEL_FILES=(
  .gitattributes
  README.md
  chat_template.json
  config.json
  generation_config.json
  merges.txt
  model-00001-of-00013.safetensors
  model-00002-of-00013.safetensors
  model-00003-of-00013.safetensors
  model-00004-of-00013.safetensors
  model-00005-of-00013.safetensors
  model-00006-of-00013.safetensors
  model-00007-of-00013.safetensors
  model-00008-of-00013.safetensors
  model-00009-of-00013.safetensors
  model-00010-of-00013.safetensors
  model-00011-of-00013.safetensors
  model-00012-of-00013.safetensors
  model-00013-of-00013.safetensors
  model.safetensors.index.json
  preprocessor_config.json
  tokenizer.json
  tokenizer_config.json
  video_preprocessor_config.json
  vocab.json
)

{
  for file in "${MODEL_FILES[@]}"; do
    printf '%s/%s\n' "$MODEL_BASE_URL" "$file"
    printf '  dir=%s\n' "$PARTIAL"
    printf '  out=%s\n' "$file"
  done
} > "$INPUT_FILE"

printf '%s\n' "${MODEL_FILES[@]}" > "$PARTIAL/files.txt"
printf '%s\n' "${MODEL_ID}@${MODEL_REVISION}" > "$PARTIAL/model_revision.txt"

"$ARIA2C" \
  --continue=true \
  --auto-file-renaming=false \
  --file-allocation=none \
  --max-concurrent-downloads=4 \
  --max-connection-per-server=8 \
  --split=8 \
  --min-split-size=20M \
  --connect-timeout=30 \
  --timeout=60 \
  --max-tries=5 \
  --retry-wait=5 \
  --summary-interval=30 \
  --input-file="$INPUT_FILE"

"$PYTHON" - "$PARTIAL" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open

target = Path(sys.argv[1])
expected_files = target.joinpath("files.txt").read_text().splitlines()
for name in expected_files:
    path = target / name
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)

index = json.loads((target / "model.safetensors.index.json").read_text())
expected_shards = {f"model-{index:05d}-of-00013.safetensors" for index in range(1, 14)}
if set(index["weight_map"].values()) != expected_shards:
    raise ValueError("weight index does not reference exactly the expected 13 shards")
for shard in sorted(expected_shards):
    with safe_open(target / shard, framework="pt") as weights:
        if not list(weights.keys()):
            raise ValueError(f"empty safetensors shard: {shard}")
print("Verified the model index and all safetensors headers")
PY

test -f "$PARTIAL/config.json"
find "$PARTIAL" -type f -name '*.safetensors' -print -quit | grep -q .
mv "$PARTIAL" "$TARGET"
echo "Downloaded ${MODEL_ID}@${MODEL_REVISION} to ${TARGET}"
