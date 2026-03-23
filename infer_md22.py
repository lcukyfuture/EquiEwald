"""
MD22 Inference script for eSCN-MD Irreps model.
Loads checkpoint and evaluates on the test set.
All parameters are extracted from the checkpoint.

Usage:
    python infer_md22.py --ckpt /path/to/best_molecule.pt
    python infer_md22.py --ckpt /path/to/best_molecule.pt --data-root ./data/MD22
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets.md22_dataset import MD22_ESCN
from fairchem.core.models.uma.escn_md_irreps import eSCNMDBackbone, MLP_EFS_Head
from train_md22 import convert_to_escn_format


def get_args():
    parser = argparse.ArgumentParser(description="MD22 Inference from Checkpoint")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint file (.pt)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override data root path (default: from checkpoint config)")
    parser.add_argument("--molecule", type=str, default=None,
                        help="Override molecule name (default: from checkpoint config)")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU index (e.g., '0')")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results (default: checkpoint directory)")
    return parser.parse_args()


def main():
    args = get_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"Using GPU: {args.gpu}")

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)

    if 'config' not in ckpt:
        raise ValueError("Checkpoint does not contain config!")

    config = ckpt['config']
    model_cfg = config['model']
    data_cfg = config['data']

    # Print checkpoint info
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  Val loss: {ckpt.get('val_loss', 'N/A'):.4f}")
    print(f"  Val energy MAE: {ckpt.get('val_mae_energy', 'N/A'):.4f}")
    print(f"  Val force MAE: {ckpt.get('val_mae_force', 'N/A'):.4f}")
    print(f"\nModel config:")
    for key, value in model_cfg.items():
        if key != 'ewald_hyperparams':
            print(f"  {key}: {value}")
    if model_cfg.get('ewald_hyperparams'):
        print(f"\nEwald config:")
        for key, value in model_cfg['ewald_hyperparams'].items():
            print(f"  {key}: {value}")

    # Extract normalization
    task_mean, task_std = config['normalization']
    print(f"\nNormalization:")
    print(f"  task_mean: {task_mean:.4f} kcal/mol")
    print(f"  task_std:  {task_std:.4f}")

    # Reconstruct model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nBuilding model on {device}...")

    backbone = eSCNMDBackbone(
        max_num_elements=model_cfg.get('num_elements', 90),
        sphere_channels=model_cfg.get('sphere_channels', 128),
        lmax=model_cfg.get('lmax', 3),
        mmax=model_cfg.get('mmax', 2),
        cutoff=model_cfg.get('cutoff', 5.0),
        max_neighbors=model_cfg.get('max_neighbors', 300),
        num_layers=model_cfg.get('num_layers', 4),
        hidden_channels=model_cfg.get('hidden_channels', 128),
        edge_channels=model_cfg.get('edge_channels', 128),
        regress_forces=True,
        direct_forces=False,
        use_pbc=False,
        always_use_pbc=False,
        otf_graph=False,
        dataset_list=[data_cfg['molecule']],
        ewald_hyperparams=model_cfg.get('ewald_hyperparams'),
    ).to(device)

    head = MLP_EFS_Head(backbone, wrap_property=True).to(device)

    # Load weights
    backbone.load_state_dict(ckpt['backbone_state_dict'])
    head.load_state_dict(ckpt['head_state_dict'])

    # Disable CUDA graph for gradient-based force computation
    if hasattr(backbone, 'use_cuda_graph_wigner'):
        backbone.use_cuda_graph_wigner = False

    backbone.eval()
    head.eval()

    total_params = sum(p.numel() for p in backbone.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"  Parameters: {total_params / 1e6:.3f}M")

    # Load test dataset
    molecule = args.molecule or data_cfg['molecule']
    data_root = args.data_root or data_cfg.get('root', './data/MD22')
    seed = data_cfg.get('seed', 42)
    print(f"\nDataset:")
    print(f"  Molecule: {molecule}")
    print(f"  Data root: {data_root}")
    print(f"  Seed: {seed}")

    test_set = MD22_ESCN(data_root, molecule, split='test', seed=seed)
    print(f"  Test samples: {len(test_set)}")

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # Inference
    print(f"\nRunning inference...")
    all_y_pred = []
    all_y_true = []
    all_dy_pred = []
    all_dy_true = []

    for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
        batch = batch.to(device)
        batch = convert_to_escn_format(batch)

        with torch.enable_grad():
            emb = backbone(batch)
            outputs = head(batch, emb)

        pred_energy = outputs['energy']['energy'].detach() + task_mean
        pred_forces = outputs['forces']['forces'].detach()

        all_y_pred.append(pred_energy.cpu())
        all_y_true.append(batch.y.cpu())
        all_dy_pred.append(pred_forces.cpu())
        all_dy_true.append(batch.force.cpu())

    # Concatenate
    y_pred = torch.cat(all_y_pred, dim=0).numpy()
    y_true = torch.cat(all_y_true, dim=0).numpy()
    dy_pred = torch.cat(all_dy_pred, dim=0).numpy()
    dy_true = torch.cat(all_dy_true, dim=0).numpy()

    # Metrics
    e_mae = np.mean(np.abs(y_pred - y_true))
    e_rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    f_mae = np.mean(np.abs(dy_pred - dy_true))
    f_rmse = np.sqrt(np.mean((dy_pred - dy_true) ** 2))

    print(f"\n{'=' * 60}")
    print(f"[Test Results - {molecule}]")
    print(f"  Energy MAE:  {e_mae:.4f} kcal/mol")
    print(f"  Energy RMSE: {e_rmse:.4f} kcal/mol")
    print(f"  Force MAE:   {f_mae:.4f} kcal/(mol·A)")
    print(f"  Force RMSE:  {f_rmse:.4f} kcal/(mol·A)")
    print(f"{'=' * 60}")

    # Output directory
    output_dir = args.output_dir or os.path.dirname(args.ckpt)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save results
    results_path = os.path.join(output_dir, "evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Molecule: {molecule}\n")
        f.write(f"Epoch: {ckpt.get('epoch', 'N/A')}\n")
        f.write(f"task_mean: {task_mean:.4f}\n")
        f.write(f"task_std: {task_std:.4f}\n\n")
        f.write(f"TEST RESULTS:\n")
        f.write(f"  Energy MAE:  {e_mae:.4f} kcal/mol\n")
        f.write(f"  Energy RMSE: {e_rmse:.4f} kcal/mol\n")
        f.write(f"  Force MAE:   {f_mae:.4f} kcal/(mol·A)\n")
        f.write(f"  Force RMSE:  {f_rmse:.4f} kcal/(mol·A)\n")
    print(f"\nResults saved to: {results_path}")

    # Save predictions
    pred_path = os.path.join(output_dir, f"test_predictions_{molecule}.npz")
    np.savez(pred_path,
             y_pred=y_pred, y_true=y_true,
             dy_pred=dy_pred, dy_true=dy_true,
             energy_mae=e_mae, energy_rmse=e_rmse,
             force_mae=f_mae, force_rmse=f_rmse)
    print(f"Predictions saved to: {pred_path}")


if __name__ == "__main__":
    main()
