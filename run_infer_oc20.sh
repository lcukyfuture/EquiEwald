#!/bin/bash
# OC20 Inference on GPU 2,3 - All 4 val splits
# Target: energy MAE ~453.0, force MAE ~38.4

export CUDA_VISIBLE_DEVICES=2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate LR

CHECKPOINT="/home/zhanglingfeng/Research/EwaldMP/checkpoints/2025-12-26-13-15-44-escn_oc20_ewald_irreps/best_checkpoint.pt"
RESULTS_DIR="./results_oc20_infer"
DATA_DIR="./data/s2ef"
BATCH_SIZE=4

mkdir -p "$RESULTS_DIR"

SPLITS=("val_id" "val_ood_ads" "val_ood_cat" "val_ood_both")

for split in "${SPLITS[@]}"; do
    echo ""
    echo "========================================"
    echo "  Running split: $split"
    echo "========================================"
    torchrun --standalone --nnodes=1 --nproc_per_node=2 infer_oc20.py \
        --checkpoint "$CHECKPOINT" \
        --split "$split" \
        --eval_batch_size "$BATCH_SIZE" \
        --data_dir "$DATA_DIR" \
        --results_dir "$RESULTS_DIR"

    if [ $? -ne 0 ]; then
        echo "ERROR: Split $split failed!"
    fi
done

# Aggregate results
echo ""
echo "========================================"
echo "  Aggregating results"
echo "========================================"
python3 -c "
import json, os, glob

results_dir = '$RESULTS_DIR'
all_results = {}
for f in sorted(glob.glob(os.path.join(results_dir, 'results_val_*.json'))):
    split = os.path.basename(f).replace('results_', '').replace('.json', '')
    with open(f) as fp:
        all_results[split] = json.load(fp)

print('=' * 60)
print('PER-SPLIT RESULTS')
print('=' * 60)
for split, res in all_results.items():
    print(f'\n{split}:')
    for k, v in res.items():
        print(f'  {k}: {v:.6f}')

if all_results:
    keys = list(next(iter(all_results.values())).keys())
    avg = {}
    for k in keys:
        vals = [r[k] for r in all_results.values() if k in r]
        avg[k] = sum(vals) / len(vals)

    print()
    print('=' * 60)
    print(f'AVERAGE ACROSS {len(all_results)} SPLITS')
    print('=' * 60)
    for k, v in avg.items():
        print(f'  {k}: {v:.6f}')

    if 'energy_mae' in avg and 'forces_mae' in avg:
        e_mae = avg['energy_mae'] * 1000
        f_mae = avg['forces_mae'] * 1000
        print(f'\n  Energy MAE: {e_mae:.1f} meV')
        print(f'  Force  MAE: {f_mae:.1f} meV/A')
        print(f'\n  Target: Energy MAE 453.0 meV, Force MAE 38.4 meV/A')

    # Save summary
    summary = {'per_split': all_results, 'average': avg}
    with open(os.path.join(results_dir, 'summary.json'), 'w') as fp:
        json.dump(summary, fp, indent=2)
    print(f'\nSummary saved to: {results_dir}/summary.json')
"
