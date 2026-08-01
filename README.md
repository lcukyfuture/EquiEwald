# EquiEwald

Ewald-based long-range message passing for equivariant graph neural networks in molecular property prediction.

## Installation

```bash
# Create conda environment
conda create -n equiewald python=3.10
conda activate equiewald

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install torch-geometric and dependencies
pip install torch-geometric torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install remaining dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Hardware

Unless otherwise specified, the experiments in this repository were conducted on NVIDIA GeForce RTX 4090 GPUs.

## Training

### AIMD-Chignolin

Train the eSCN-MD model on the AIMD-Chignolin protein dataset (PyTorch Lightning):

```bash
# Single GPU
python train_chig.py --config configs/chig/chig_escn.yaml

# Multi-GPU
python train_chig.py --config configs/chig/chig_escn.yaml --gpus 0,1

# Enable Ewald blocks
python train_chig.py --irreps
```

### Dimer

Train the eSCN-MD Irreps model on the charged dimer interaction energy/forces dataset:

```bash
# Single GPU
python train_dimer.py --config configs/dimer/escn_md_irreps.yaml

```

### Molten NaCl

Train the periodic eSCN-MD Irreps model on molten NaCl:

```bash
python train_nacl.py --config configs/nacl/escn_md_irreps_pbc.yaml
```

Evaluate a saved checkpoint on the deterministic validation split:

```bash
python evaluate_nacl.py \
  --checkpoint /path/to/model-best.pth \
  --config configs/nacl/escn_md_irreps_pbc.yaml
```

### OC20

Train the eSCN model on the OC20 catalysis dataset:

```bash
# Single GPU
python train_oc20.py

# Multi-GPU (via torchrun)
torchrun --nproc_per_node=2 train_oc20.py
```

## Data Setup

### AIMD-Chignolin

Place the AIMD-Chignolin dataset under `./data/smaller_Chig_AIMD/` with the following structure:
```
data/smaller_Chig_AIMD/
├── Coordinates/
├── Force/
└── Split/
    └── scaffold.npz
```

### Molten NaCl

Place `ML_AB_NaCl_liquid_Extract.xyz` under:

```text
data/nacl/raw/ML_AB_NaCl_liquid_Extract.xyz
```

Each structure must provide a periodic cell, energy, and forces.

### OC20

Download the OC20 S2EF dataset and place it under `./data/s2ef/`. See [OC20 documentation](https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md) for details.

## Project Structure

```
EquiEwald/
├── train_chig.py          # AIMD-Chig training script
├── train_dimer.py         # Dimer training script
├── train_nacl.py          # Periodic molten-NaCl training script
├── evaluate_nacl.py       # Molten-NaCl checkpoint evaluation
├── train_oc20.py          # OC20 training script
├── configs/               # Configuration files
├── datasets/              # Dataset loaders
├── fairchem/              # eSCN-MD backbone (Fairchem UMA)
└── ocpmodels/             # OC20 model framework
```
