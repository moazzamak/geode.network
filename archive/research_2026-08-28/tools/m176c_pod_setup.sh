#!/usr/bin/env bash
# M176c pod setup (frozen 17 Aug 2026). Runs ON the rental. Registered in
# analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md §12 before first execution.
# Stage 0 only: probe, environment, data with integrity checks. No fits.
set -euo pipefail

REPO=/workspace/CG-MoE
CACHE=/workspace/geode_cache
# M176c stage-0 repair #2 (registered 17 Aug): venv and the HF download
# cache live on the LOCAL overlay disk — pip over the mfs network volume
# stalls in uninterruptible I/O (measured: pip D-state, 0 progress). Only
# the GEODE data layout lives on /workspace. Local overlay budget: venv
# ~2 GB + HF cache ~18.5 GB (deleted after verification) < 30 GB disk.
VENV_DIR=/root/.venv-pod
HF_HOME_DIR=/root/hf_cache
HF_REPO=wltjr1007/DomainNet
HF_REV=ee20570ae7a29c51571e55a9a17983f7625295d6

# Registered parquet integrity constants (from the local manifest.json)
P0="7eb34c7c9c020f265db6c4b2405c873f4bda0259cd06b43aa31df45a17a55409 test-00000-of-00001.parquet"
P1="37dfda4256254a53d58352ba6f3ea8a1ae24d13f3d39eb27a143b859f73b3e5a train-00000-of-00003.parquet"
P2="c5d86606a2fa7b1418895717803cfa7c3e7adad37454ee7b1ad44f1ec0eb3e15 train-00001-of-00003.parquet"
P3="bb3dd680e02ac1cf539fb6d32959b096a06535892aa8c494e31378762f02613c train-00002-of-00003.parquet"

REPORT="$REPO/logs/pod_report.txt"
mkdir -p "$REPO/logs"

echo "== pod facts ==" | tee "$REPORT"
date -u | tee -a "$REPORT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  | tee -a "$REPORT"
df -h / | tail -1 | tee -a "$REPORT"
df -h /workspace | tail -1 | tee -a "$REPORT"
free -h | head -2 | tee -a "$REPORT"
python --version | tee -a "$REPORT"
# the bundle is a git archive (no .git); the commit travels in RELEASE_COMMIT
if [ -f "$REPO/RELEASE_COMMIT" ]; then
  echo "repo commit: $(cat "$REPO/RELEASE_COMMIT")" | tee -a "$REPORT"
else
  echo "repo commit: UNKNOWN (RELEASE_COMMIT missing)" | tee -a "$REPORT"
fi

echo "== environment ==" | tee -a "$REPORT"
python -m venv --system-site-packages "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install -q numpy scikit-learn scikit-image datasets Pillow kagglehub
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  | tee -a "$REPORT" || {
  echo "image has no torch; installing cu124 wheels" | tee -a "$REPORT"
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
    | tee -a "$REPORT"
}

echo "== data download (HF, pinned revision, direct parquet files) ==" | tee -a "$REPORT"
# M176c stage-0 repair #3 (registered 17 Aug): load_dataset() double-buffers
# (parquet download + arrow conversion) and filled the 30 GB local disk
# (Errno 28). Download the four parquets directly; no arrow cache exists.
# Local budget: ~18.5 GB downloads (deleted after verify) + ~2 GB venv.
export HF_HOME="$HF_HOME_DIR"
DL_DIR="$HF_HOME_DIR/data"
mkdir -p "$DL_DIR"
BASE_URL="https://huggingface.co/datasets/$HF_REPO/resolve/$HF_REV/data"
for pair in "$P0" "$P1" "$P2" "$P3"; do
  want=${pair%% *}; name=${pair##* }
  echo "downloading $name" | tee -a "$REPORT"
  curl -fL --retry 3 --retry-delay 2 -o "$DL_DIR/$name" "$BASE_URL/$name"
done

echo "== integrity: sha256 of the four parquets ==" | tee -a "$REPORT"
TARGET="$CACHE/domainnet/repository/data"
mkdir -p "$TARGET"
for pair in "$P0" "$P1" "$P2" "$P3"; do
  want=${pair%% *}; name=${pair##* }
  src="$DL_DIR/$name"
  got=$(sha256sum "$src" | cut -d' ' -f1)
  if [ "$got" != "$want" ]; then
    echo "SHA MISMATCH $name: got $got want $want" | tee -a "$REPORT"
    exit 1
  fi
  echo "ok $name" | tee -a "$REPORT"
  cp "$src" "$TARGET/$name"
done

echo "== manifest (pinned, written verbatim) ==" | tee -a "$REPORT"
cat > "$CACHE/domainnet/manifest.json" <<'MANIFEST'
{
  "class_count": 345,
  "domains": ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
  "files": [
    {"path": "data/test-00000-of-00001.parquet",
     "sha256": "7eb34c7c9c020f265db6c4b2405c873f4bda0259cd06b43aa31df45a17a55409",
     "size": 5597563518},
    {"path": "data/train-00000-of-00003.parquet",
     "sha256": "37dfda4256254a53d58352ba6f3ea8a1ae24d13f3d39eb27a143b859f73b3e5a",
     "size": 758577202},
    {"path": "data/train-00001-of-00003.parquet",
     "sha256": "c5d86606a2fa7b1418895717803cfa7c3e7adad37454ee7b1ad44f1ec0eb3e15",
     "size": 7205696451},
    {"path": "data/train-00002-of-00003.parquet",
     "sha256": "bb3dd680e02ac1cf539fb6d32959b096a06535892aa8c494e31378762f02613c",
     "size": 4959599036}
  ],
  "schema_version": 2,
  "source_repository": "wltjr1007/DomainNet",
  "source_revision": "ee20570ae7a29c51571e55a9a17983f7625295d6",
  "split_samples": {"test": 176743, "train": 409832},
  "version": "huggingface-parquet-v1"
}
MANIFEST

# Disk discipline (registered): drop the HF download cache now that the
# parquets sit in the GEODE layout with verified hashes.
rm -rf "$HF_HOME_DIR"
df -h / | tail -1 | tee -a "$REPORT"
echo "SETUP COMPLETE" | tee -a "$REPORT"
