#!/usr/bin/env bash
# Evaluate all four OC20 validation splits in the active Python environment.
set -euo pipefail
CHECKPOINT="${1:?Usage: bash run_infer_oc20.sh CHECKPOINT [DATA_DIR] [RESULTS_DIR]}"
DATA_DIR="${2:-./data/s2ef}"
RESULTS_DIR="${3:-./results_oc20_infer}"
NPROC="${NPROC:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
mkdir -p "$RESULTS_DIR"
for split in val_id val_ood_ads val_ood_cat val_ood_both; do
    torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" infer_oc20.py \
        --checkpoint "$CHECKPOINT" --split "$split" \
        --eval_batch_size "$BATCH_SIZE" --data_dir "$DATA_DIR" \
        --results_dir "$RESULTS_DIR"
done
python - "$RESULTS_DIR" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
splits = ('val_id', 'val_ood_ads', 'val_ood_cat', 'val_ood_both')
results = {split: json.loads((root / f'results_{split}.json').read_text()) for split in splits}
keys = set.intersection(*(set(r) for r in results.values()))
average = {key: sum(r[key] for r in results.values()) / len(results) for key in sorted(keys)}
summary = {'per_split': results, 'unweighted_split_average': average}
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
PY
