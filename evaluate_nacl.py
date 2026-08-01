#!/usr/bin/env python3
"""Evaluate a periodic NaCl eSCN-irreps checkpoint on its validation split."""

from __future__ import annotations

import argparse

import torch
from torch_geometric.loader import DataLoader

from train_nacl import (
    build_model,
    default_config,
    load_config,
    load_dataset,
    merge_config,
    run_epoch,
    set_seed,
    split_dataset,
    torch_load,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint produced by train_nacl.py.")
    parser.add_argument("--config", default="configs/nacl/escn_md_irreps_pbc.yaml")
    parser.add_argument("--data-path", default=None, help="Override data_path from the YAML config.")
    parser.add_argument("--device", default=None, help="Torch device, for example cuda:0 or cpu.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = default_config()
    merge_config(config, load_config(args.config))
    if args.data_path is not None:
        config["data_path"] = args.data_path

    set_seed(int(config["seed"]))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = load_dataset(config["data_path"])
    _, valid_set = split_dataset(dataset, float(config["valid_fraction"]), int(config["seed"]))
    valid_loader = DataLoader(valid_set, batch_size=int(config["batch_size"]), shuffle=False)

    model = build_model(config, dataset, device)
    checkpoint = torch_load(args.checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = run_epoch(
        model,
        valid_loader,
        device,
        optimizer=None,
        energy_weight=float(config["energy_weight"]),
        force_weight=float(config["force_weight"]),
        grad_clip_norm=float(config["grad_clip_norm"]),
        desc="Validation",
    )

    checkpoint_epoch = int(checkpoint.get("epoch", -1)) + 1
    print(f"checkpoint_epoch: {checkpoint_epoch}")
    print(f"validation_frames: {len(valid_set)}")
    print(f"energy_rmse_meV_per_atom: {metrics['energy_rmse'] * 1000:.6f}")
    print(f"energy_mae_meV_per_atom: {metrics['energy_mae'] * 1000:.6f}")
    print(f"force_rmse_meV_per_A: {metrics['force_rmse'] * 1000:.6f}")
    print(f"force_mae_meV_per_A: {metrics['force_mae'] * 1000:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
