# EquiEwald

Ewald-based long-range message passing for equivariant graph neural networks.
This repository contains nonperiodic eSCN-MD models, an OC20 eSCN model, and a
standalone periodic molten-NaCl model.

This is a code release for anonymous peer review. See
[VALIDATION.md](VALIDATION.md) for the checks actually run and their limitations.
Small smoke tests establish that the implementation executes; they do not
establish the accuracy reported in the paper.

## Installation

Use a fresh Python 3.10–3.12 environment. Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install setuptools==75.8.0 wheel==0.45.1
```

The PyTorch version and PyG extension wheel index must match; do not mix
extensions from a different Torch/CUDA build.

**NVIDIA CUDA 12.1 (Linux; installation recipe, not locally verified):**

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install torch-scatter==2.1.2 torch-cluster==1.6.3 \
  -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

Then install the pinned runtime and test dependencies:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install --no-build-isolation -e .
python -m pip check
```

The vendored `fairchem`, `ocpmodels`, and `datasets` packages are included in the
editable installation. Do not install another package providing these names in
the same environment. Optional legacy preprocessing/HPO utilities may require
`pymatgen`/`ray`; neither is needed by the smoke tests. The release does not
require `torchvision` or the optional `fairchem_cpp` kernels.

## Training and evaluation

### Dimer

The default configuration uses the included `id5` XYZ files: 10 training
structures and 3 validation structures, each with 24 atoms. The file named
`test-id5.xyz` is used as the validation input by the supplied training script;
it is **not an independent held-out test** for model selection. Supply a
separate validation split for a protocol requiring independent test evaluation.

```bash
python train_dimer.py --config configs/dimer/escn_md_irreps.yaml
```

Input: extended XYZ in `data/dimer/raw/`, with `inter_energy` and per-atom
`forces`. ASE reads the force calculator fields. Numerical units are retained
from the source files. Do not assume units or fragment-force conventions from
the filename alone. The supplied head obtains forces by differentiating energy.

The default configuration addresses **id5 only**, not every Dimer experiment.
Other systems require their corresponding files and configuration. To reload and evaluate a checkpoint on its configured validation file:

```bash
python train_dimer.py --mode test --checkpoint /path/to/best_dimer.pt
```

The architecture is restored from the checkpoint. Use `--root` to relocate the
data, or `--override` with `data.val_xyz` to select a separate evaluation file.
Load checkpoints only from a trusted source.

### AIMD-Chignolin

Provide:

```text
data/smaller_Chig_AIMD/
├── Coordinates/
├── Force/
└── Split/scaffold.npz
```

```bash
# Baseline (irreps is false in the default YAML)
python train_chig.py --config configs/chig/chig_escn.yaml
# Enable long-range Ewald blocks
python train_chig.py --config configs/chig/chig_escn.yaml --irreps
# Multi-GPU, when supported by the environment
python train_chig.py --config configs/chig/chig_escn.yaml --irreps --gpus 0,1
python infer_chig.py --ckpt /path/to/model.ckpt --data-root ./data/smaller_Chig_AIMD
```

The loader converts Hartree energies and Hartree/Å forces to kcal/mol and
kcal/(mol·Å). The dataset and pretrained checkpoints are not bundled.

### Molten NaCl (periodic)

Provide `data/nacl/raw/ML_AB_NaCl_liquid_Extract.xyz`, with periodic cell,
energy, and forces for every structure.

```bash
python train_nacl.py --config configs/nacl/escn_md_irreps_pbc.yaml
python train_nacl.py --config configs/nacl/escn_md_irreps_pbc.yaml --smoke --max-frames 8
python evaluate_nacl.py --checkpoint /path/to/model-best.pth \
  --config configs/nacl/escn_md_irreps_pbc.yaml
```

The dataset is not bundled. Automated tests use explicitly synthetic periodic
structures and do not evaluate molten-NaCl prediction accuracy.

### OC20

Obtain the S2EF data from the
[Open Catalyst Project](https://opencatalystproject.org/), arrange the LMDB
splits under `data/s2ef/`, and check all paths in
`configs/oc20/escn_ewald.yml`. The example configuration uses validation
subsampling; inspect this setting before reproducing a full benchmark.

```bash
python train_oc20.py --config configs/oc20/escn_ewald.yml
# Or launch one process per GPU:
torchrun --standalone --nproc_per_node=2 train_oc20.py
python infer_oc20.py --checkpoint /path/to/best_checkpoint.pt \
  --split val_id --data_dir ./data/s2ef
# All four splits; uses the current Python environment:
NPROC=2 bash run_infer_oc20.sh /path/to/best_checkpoint.pt ./data/s2ef
```

OC20 training uses CUDA/NCCL. The shell wrapper stops on failure and records the
per-split results plus their unweighted arithmetic average. No author-specific
checkpoint path or Conda environment is assumed. OC20 data and checkpoints are
not bundled.

## Layout

```text
configs/             Experiment and smoke-test configurations
datasets/            Dimer and Chignolin readers
fairchem/            eSCN-MD backbone and long-range blocks
ocpmodels/           OC20 framework and eSCN model
data/dimer/raw/      Included id5 example data
experiments_data/    Previously supplied plotting tables (not regenerated here)
train_*.py           Training entry points
infer_*.py           Chignolin and OC20 evaluation entry points
evaluate_nacl.py     Periodic checkpoint evaluation
tests/               Execution and regression checks
validation/          Verification environment record
LICENSES/            Licenses for bundled upstream code
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution.
Names in those notices identify third-party code, not authors of this paper.
