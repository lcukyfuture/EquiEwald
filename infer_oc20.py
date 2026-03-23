"""
OC20 Model Inference / Validation Script
Load a checkpoint and evaluate on a single validation split.
Run with torchrun for multi-GPU support.
"""

import os
import json
import gc
import argparse
import torch
import torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", 0))
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

import torch_geometric
import logging
from pathlib import Path

from ocpmodels import models
from ocpmodels.common import logger
from ocpmodels.common.utils import setup_logging, load_config
from ocpmodels.datasets import LmdbDataset
from ocpmodels.common.registry import registry
from ocpmodels.trainers import ForcesTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="OC20 Inference / Validation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/zhanglingfeng/Research/EwaldMP/checkpoints/"
                "2025-12-26-13-15-44-escn_oc20_ewald_irreps/best_checkpoint.pt",
        help="Path to the checkpoint file (.pt)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val_id",
        choices=["val_id", "val_ood_ads", "val_ood_cat", "val_ood_both"],
        help="Validation split to evaluate",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/s2ef",
        help="Base directory for OC20 S2EF data",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=5000,
        help="Print frequency during validation",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory to save per-split JSON results (default: checkpoint dir)",
    )
    return parser.parse_args()


def load_checkpoint_config(checkpoint_path):
    """Load configuration from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" in checkpoint:
        config = checkpoint["config"]
        print("Successfully loaded config from checkpoint")
        return config
    else:
        raise ValueError("Checkpoint does not contain 'config' key.")


def get_dataset_config(split, data_dir, train_config):
    """Build dataset config for a specific validation split."""
    if isinstance(train_config, list):
        train_cfg = train_config[0]
    else:
        train_cfg = train_config

    base_train_config = {
        "src": os.path.join(data_dir, "2M/train"),
        "normalize_labels": train_cfg.get("normalize_labels", True),
        "target_mean": train_cfg.get("target_mean", -0.7554450631141663),
        "target_std": train_cfg.get("target_std", 2.887317180633545),
        "grad_target_mean": train_cfg.get("grad_target_mean", 0.0),
        "grad_target_std": train_cfg.get("grad_target_std", 2.887317180633545),
    }

    split_paths = {
        "val_id": "all/val_id",
        "val_ood_ads": "all/val_ood_ads",
        "val_ood_cat": "all/val_ood_cat",
        "val_ood_both": "all/val_ood_both",
    }
    val_config = {"src": os.path.join(data_dir, split_paths[split])}
    return [base_train_config, val_config]


def main():
    args = parse_args()

    # Initialize distributed
    if not dist.is_initialized():
        if "RANK" not in os.environ:
            os.environ["RANK"] = str(local_rank)
        if "WORLD_SIZE" not in os.environ:
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            world_size = len(cuda_visible.split(",")) if cuda_visible else 1
            os.environ["WORLD_SIZE"] = str(world_size)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ["RANK"]),
        )

    setup_logging()

    print(f"[Info] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[Info] LOCAL_RANK: {local_rank}")
    if torch.cuda.is_available():
        print(f"[Info] GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # Load config from checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    config = load_checkpoint_config(args.checkpoint)

    # Extract configs
    task = config.get("task", {})
    model_config = config.get("model", {})
    if isinstance(model_config, str):
        model_attrs = config.get("model_attributes", {})
        model_config = {"name": model_config, **model_attrs}

    optimizer = config.get("optim", config.get("optimizer", {}))
    if isinstance(optimizer, dict):
        optimizer["eval_batch_size"] = args.eval_batch_size

    train_dataset_config = config.get("dataset", config.get("dataset_train", [{}]))
    dataset_config = get_dataset_config(args.split, args.data_dir, train_dataset_config)

    print(f"\n{'=' * 60}")
    print(f"Validating on split: {args.split}")
    print(f"{'=' * 60}")

    # Create validator
    validator = ForcesTrainer(
        task=task,
        model=model_config,
        dataset=dataset_config,
        optimizer=optimizer,
        identifier=f"validate_{args.split}",
        run_dir="./",
        is_debug=True,
        print_every=args.print_every,
        seed=0,
        logger="tensorboard",
        local_rank=local_rank,
        amp=args.amp,
    )

    validator.load_checkpoint(checkpoint_path=args.checkpoint)
    print(f"Running validation on {args.split}...")
    metrics = validator.validate()

    results = {key: float(val["metric"]) for key, val in metrics.items()}

    print(f"\n{args.split} Results:")
    print("-" * 40)
    for key, value in results.items():
        print(f"  {key}: {value:.6f}")

    # Save per-split results as JSON (only rank 0)
    if local_rank == 0:
        results_dir = args.results_dir or str(Path(args.checkpoint).parent)
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f"results_{args.split}.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_file}")

    return results


if __name__ == "__main__":
    main()
