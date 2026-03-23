"""
AIMD-Chig Inference script for eSCN-MD model.
Loads checkpoint and evaluates on validation/test sets.
All parameters are extracted from the checkpoint.

Usage:
    python infer_chig.py --ckpt /path/to/checkpoint.ckpt
    python infer_chig.py --ckpt /path/to/checkpoint.ckpt --splits vali_id test_id
    python infer_chig.py --ckpt /path/to/checkpoint.ckpt --data-root ./data/smaller_Chig_AIMD
"""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from train_chig import ChigESCNLightningModule
from datasets.chig_dataset import AIMDChigDataset


def get_args():
    parser = argparse.ArgumentParser(description="AIMD-Chig Inference from Checkpoint")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint file (.ckpt)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override data root path (default: from checkpoint config)")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU index (e.g., '0')")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results (default: checkpoint directory)")
    parser.add_argument("--splits", type=str, nargs='+', default=['test_id'],
                        choices=['train_id', 'vali_id', 'test_id'],
                        help="Which splits to evaluate (default: test_id)")
    return parser.parse_args()


def evaluate_split(model, dataset, batch_size, num_workers, split_name):
    """Evaluate model on a dataset split with gradient-based force computation."""
    print(f"\nEvaluating {split_name}...")
    print(f"  Dataset size: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )

    model.eval()
    device = next(model.parameters()).device

    all_y_pred = []
    all_y_true = []
    all_dy_pred = []
    all_dy_true = []

    for batch in tqdm(dataloader, desc=f"Evaluating {split_name}", unit="batch"):
        batch = batch.to(device)

        # Gradients required for force computation via torch.autograd.grad
        with torch.enable_grad():
            pred_energy, pred_forces, data = model(batch)

        # Denormalize energy (model predicts E - task_mean)
        pred_energy = pred_energy.detach() + model.task_mean

        all_y_pred.append(pred_energy.cpu())
        all_y_true.append(data.y.cpu())
        all_dy_pred.append(pred_forces.detach().cpu())
        all_dy_true.append(data.force.cpu())

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

    return {
        'energy_mae': e_mae,
        'energy_rmse': e_rmse,
        'force_mae': f_mae,
        'force_rmse': f_rmse,
        'y_pred': y_pred,
        'y_true': y_true,
        'dy_pred': dy_pred,
        'dy_true': dy_true,
    }


def main():
    args = get_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"Using GPU: {args.gpu}")

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    # Load checkpoint to extract hyperparameters
    print(f"\nLoading checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)

    if 'hyper_parameters' not in checkpoint:
        raise ValueError("Checkpoint does not contain hyperparameters!")

    config = checkpoint['hyper_parameters']
    state_dict = checkpoint['state_dict']

    # Print checkpoint info
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Global step: {checkpoint.get('global_step', 'N/A')}")
    print(f"\nModel config:")
    for key, value in config.get('model', {}).items():
        print(f"  {key}: {value}")
    if config.get('ewald_hyperparams'):
        print(f"\nEwald config:")
        for key, value in config['ewald_hyperparams'].items():
            print(f"  {key}: {value}")

    # Extract normalization parameters from state_dict
    task_mean = state_dict['task_mean'].item() if 'task_mean' in state_dict else 0.0
    task_std = state_dict['task_std'].item() if 'task_std' in state_dict else 1.0
    print(f"\nNormalization:")
    print(f"  task_mean: {task_mean:.4f} kcal/mol")
    print(f"  task_std:  {task_std:.4f}")

    # Reconstruct model from checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nLoading model on {device}...")

    model = ChigESCNLightningModule.load_from_checkpoint(
        args.ckpt,
        map_location=device,
        config=config,
        task_mean=task_mean,
        task_std=task_std,
    )

    # Disable CUDA graph for gradient-based force computation
    if hasattr(model.backbone, 'use_cuda_graph_wigner'):
        model.backbone.use_cuda_graph_wigner = False

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params / 1e6:.3f}M")

    # Resolve data path
    data_root = args.data_root or config.get('data', {}).get('root', './data/smaller_Chig_AIMD')
    split_npz = config.get('data', {}).get('split_npz', 'scaffold.npz')
    print(f"\nData root: {data_root}")
    print(f"Split file: {split_npz}")

    # Output directory
    output_dir = args.output_dir or os.path.dirname(args.ckpt)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Evaluate each split
    all_results = {}

    for split_key in args.splits:
        print("\n" + "=" * 60)
        dataset = AIMDChigDataset(
            data_root,
            split_npz=split_npz,
            split_key=split_key,
            validate_data=False,
            use_cache=True,
        )

        results = evaluate_split(model, dataset, args.batch_size, args.num_workers, split_key)
        all_results[split_key] = results

        print(f"\n[{split_key} Results]")
        print(f"  Energy MAE:  {results['energy_mae']:.4f} kcal/mol")
        print(f"  Energy RMSE: {results['energy_rmse']:.4f} kcal/mol")
        print(f"  Force MAE:   {results['force_mae']:.4f} kcal/(mol·A)")
        print(f"  Force RMSE:  {results['force_rmse']:.4f} kcal/(mol·A)")

    # Save results
    results_path = os.path.join(output_dir, "evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"task_mean: {task_mean:.4f} kcal/mol\n\n")
        for split_key, results in all_results.items():
            f.write(f"{split_key.upper()}:\n")
            f.write(f"  Energy MAE:  {results['energy_mae']:.4f} kcal/mol\n")
            f.write(f"  Energy RMSE: {results['energy_rmse']:.4f} kcal/mol\n")
            f.write(f"  Force MAE:   {results['force_mae']:.4f} kcal/(mol·A)\n")
            f.write(f"  Force RMSE:  {results['force_rmse']:.4f} kcal/(mol·A)\n\n")
    print(f"\nResults saved to: {results_path}")

    # Save predictions
    for split_key, results in all_results.items():
        pred_path = os.path.join(output_dir, f"{split_key}_predictions.npz")
        np.savez(pred_path,
                 y_pred=results['y_pred'], y_true=results['y_true'],
                 dy_pred=results['dy_pred'], dy_true=results['dy_true'],
                 energy_mae=results['energy_mae'], energy_rmse=results['energy_rmse'],
                 force_mae=results['force_mae'], force_rmse=results['force_rmse'])
        print(f"Predictions saved to: {pred_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for split_key, results in all_results.items():
        print(f"  {split_key}: E-MAE={results['energy_mae']:.4f} kcal/mol, "
              f"F-MAE={results['force_mae']:.4f} kcal/(mol·A)")
    print("=" * 60)


if __name__ == "__main__":
    main()
