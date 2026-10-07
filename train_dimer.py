#!/usr/bin/env python3
"""
Dimer training script for eSCN-MD model with Multi-GPU support.

Trains on charged dimer interaction energies and forces.
Supports MSE/MAE loss, StepLR/CosineAnnealing scheduler, LR warmup,
per-atom RMSE/MAE metrics, and top-K checkpoint saving.
"""
import os
import sys
import math
import time
import argparse
from pathlib import Path
from typing import Iterable, Dict
import yaml
import logging

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from datasets.dimer_dataset import DimerDataset
from fairchem.core.models.uma.escn_md_irreps import eSCNMDBackbone, MLP_EFS_Head


def deep_update(base_dict, override_dict):
    """Recursively merge override_dict into base_dict."""
    for key, value in override_dict.items():
        if (
            key in base_dict
            and isinstance(base_dict[key], dict)
            and isinstance(value, dict)
        ):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def setup_logging(log_dir, run_name, rank=0):
    """Setup logging to both file and console."""
    log_file = os.path.join(log_dir, f"{run_name}_training_rank{rank}.log")

    logger = logging.getLogger(f"rank{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    if rank == 0:
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger, log_file


def setup_distributed(rank, world_size, backend='nccl'):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum, self.cnt = 0.0, 0
    @property
    def avg(self):
        return self.sum / max(1, self.cnt)
    def update(self, val, n=1):
        self.sum += float(val) * int(n)
        self.cnt += int(n)


def get_args():
    parser = argparse.ArgumentParser("Dimer + eSCN-MD Multi-GPU Training")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="configs/dimer/escn_md_irreps.yaml")
    parser.add_argument("--override", type=str, default=None, help="Extra YAML to override config")
    parser.add_argument("--root", type=str, default=None, help="Override data root directory")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--loss-type", type=str, default=None, choices=["mse", "mae"])
    parser.add_argument("--use-ewald", action="store_true", help="Enable Ewald blocks")
    parser.add_argument("--no-ewald", action="store_true", help="Disable Ewald blocks")
    parser.add_argument("--gpus", type=str, default=None, help="GPU IDs (comma-separated)")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_args_with_config(args, config):
    """Merge command line arguments with config."""
    if args.root is not None:
        config['data']['root'] = args.root
    if args.seed is not None:
        config['data']['seed'] = args.seed

    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.eval_batch_size is not None:
        config['training']['eval_batch_size'] = args.eval_batch_size
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.loss_type is not None:
        config['training']['loss_type'] = args.loss_type

    if args.use_ewald:
        config['model']['ewald_hyperparams'] = config['model'].get('ewald_hyperparams', {
            "k_cutoff": 0.6,
            "delta_k": 0.2,
            "num_k_rbf": 128,
            "downprojection": 8,
            "num_hidden": 1,
        })
    if args.no_ewald:
        config['model']['ewald_hyperparams'] = None

    if args.gpus is not None:
        gpu_list = [int(x.strip()) for x in args.gpus.split(',')]
        config['system']['gpus'] = gpu_list

    # Override with extra YAML if provided
    if args.override:
        with open(args.override, 'r') as f:
            override = yaml.safe_load(f) or {}
        deep_update(config, override)

    return config


def count_params(model: torch.nn.Module, print_top_k: int = 15, logger=None):
    """Count model parameters."""
    if isinstance(model, DDP):
        model = model.module

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log_fn = logger.info if logger else print
    log_fn(f"[Params] total={total/1e6:.3f}M, trainable={trainable/1e6:.3f}M")

    bucket = []
    for name, m in model.named_children():
        n = sum(p.numel() for p in m.parameters())
        bucket.append((name, n))
    bucket.sort(key=lambda x: x[1], reverse=True)
    for name, n in bucket[:print_top_k]:
        log_fn(f"  - {name:<24} {n/1e6:>7.3f}M")


def _log_step_scalars(writer: SummaryWriter, split: str, scalars: Dict[str, float], step: int):
    """Write scalars to TensorBoard."""
    if writer is None:
        return
    for k, v in scalars.items():
        writer.add_scalar(f"{split}/{k}", float(v), step)


def convert_to_escn_format(data):
    """Ensure data has all required fields for eSCN-MD."""
    if not hasattr(data, 'natoms'):
        data.natoms = torch.bincount(data.batch)

    if not hasattr(data, 'cell'):
        batch_size = int(data.batch.max()) + 1
        data.cell = torch.eye(3, device=data.pos.device, dtype=data.pos.dtype)
        data.cell = data.cell.unsqueeze(0).expand(batch_size, 3, 3).contiguous() * 100.0

    if not hasattr(data, 'charge'):
        batch_size = int(data.batch.max()) + 1
        data.charge = torch.zeros(batch_size, device=data.pos.device, dtype=data.pos.dtype)

    if not hasattr(data, 'spin'):
        batch_size = int(data.batch.max()) + 1
        data.spin = torch.zeros(batch_size, device=data.pos.device, dtype=data.pos.dtype)

    if not hasattr(data, 'pbc'):
        batch_size = int(data.batch.max()) + 1
        data.pbc = torch.zeros(batch_size, 3, dtype=torch.bool, device=data.pos.device)

    return data


def train_one_epoch(
    config,
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    energy_criterion: nn.Module,
    force_criterion: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter = None,
    global_step: int = 0,
    print_freq: int = 100,
    rank: int = 0,
    world_size: int = 1,
    warmup_scheduler=None,
    lr_warmup_steps: int = 0,
):
    backbone.train()
    head.train()

    loss_meters = {"energy": AverageMeter(), "force": AverageMeter(), "total": AverageMeter()}
    mae_meters = {"energy": AverageMeter(), "force": AverageMeter()}
    mse_meters = {"energy": AverageMeter(), "force": AverageMeter()}

    train_cfg = config['training']
    t0 = time.perf_counter()

    for step, data in enumerate(data_loader):
        data = data.to(device)
        data = convert_to_escn_format(data)

        emb = backbone(data)
        outputs = head(data, emb)

        pred_energy = outputs['energy']['energy']
        pred_forces = outputs['forces']['forces']

        if torch.isnan(pred_energy).any() or torch.isnan(pred_forces).any():
            if rank == 0:
                print(f"WARNING: NaN detected! Skipping batch.")
            continue

        loss_e = energy_criterion(pred_energy, data.y)
        loss_f = force_criterion(pred_forces, data.force)
        loss = train_cfg['energy_weight'] * loss_e + train_cfg['force_weight'] * loss_f

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg['clip_grad'] and train_cfg['clip_grad'] > 0:
            if world_size > 1:
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.module.parameters()) + list(head.module.parameters()),
                    train_cfg['clip_grad']
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(head.parameters()),
                    train_cfg['clip_grad']
                )
        optimizer.step()

        # Step warmup scheduler
        if warmup_scheduler is not None and global_step < lr_warmup_steps:
            warmup_scheduler.step()

        # Compute per-atom metrics (matching source train_escn_dimer.py)
        with torch.no_grad():
            natoms = data.natoms.float().view(-1)
            e_mae = torch.mean(torch.abs(data.y / natoms - pred_energy.detach() / natoms)).item()
            f_mae = torch.mean(torch.abs(data.force - pred_forces.detach())).item()
            e_mse = torch.mean((data.y / natoms - pred_energy.detach() / natoms) ** 2).item()
            f_mse = torch.mean((data.force - pred_forces.detach()) ** 2).item()

        bs = pred_energy.shape[0]
        loss_meters["energy"].update(loss_e.item(), n=bs)
        loss_meters["force"].update(loss_f.item(), n=pred_forces.shape[0])
        loss_meters["total"].update(loss.item(), n=bs)
        mae_meters["energy"].update(e_mae, n=bs)
        mae_meters["force"].update(f_mae, n=pred_forces.shape[0])
        mse_meters["energy"].update(e_mse, n=bs)
        mse_meters["force"].update(f_mse, n=pred_forces.shape[0])

        if rank == 0:
            _log_step_scalars(writer, "train", {
                "loss_energy": loss_e.item(),
                "loss_force": loss_f.item(),
                "loss_total": loss.item(),
                "e_mae": e_mae,
                "f_mae": f_mae,
                "e_rmse": math.sqrt(e_mse),
                "f_rmse": math.sqrt(f_mse),
                "lr": optimizer.param_groups[0]["lr"],
            }, global_step)

        if rank == 0 and ((step % print_freq == 0) or (step == len(data_loader) - 1)):
            elapsed = time.perf_counter() - t0
            speed_ms = 1e3 * elapsed / max(1, (step + 1))
            e_rmse = math.sqrt(mse_meters["energy"].avg)
            f_rmse = math.sqrt(mse_meters["force"].avg)
            print(
                f"Epoch[{epoch}] Step[{step}/{len(data_loader)}] "
                f"loss_e={loss_meters['energy'].avg:.5f} "
                f"loss_f={loss_meters['force'].avg:.5f} "
                f"loss_total={loss_meters['total'].avg:.5f} "
                f"e_MAE={mae_meters['energy'].avg:.5f} "
                f"f_MAE={mae_meters['force'].avg:.5f} "
                f"e_RMSE={e_rmse:.5f} "
                f"f_RMSE={f_rmse:.5f} "
                f"{speed_ms:.1f}ms/step"
            )

        global_step += 1

    if rank == 0:
        e_rmse = math.sqrt(mse_meters["energy"].avg)
        f_rmse = math.sqrt(mse_meters["force"].avg)
        _log_step_scalars(writer, "train", {
            "epoch_loss_energy": loss_meters["energy"].avg,
            "epoch_loss_force": loss_meters["force"].avg,
            "epoch_loss_total": loss_meters["total"].avg,
            "epoch_e_mae": mae_meters["energy"].avg,
            "epoch_f_mae": mae_meters["force"].avg,
            "epoch_e_rmse": e_rmse,
            "epoch_f_rmse": f_rmse,
        }, epoch)

    return mae_meters, mse_meters, loss_meters, global_step


@torch.no_grad()
def evaluate(
    config,
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    energy_criterion: nn.Module,
    force_criterion: nn.Module,
    data_loader: Iterable,
    device: torch.device,
    writer: SummaryWriter = None,
    split_name: str = "val",
    global_step: int = 0,
    print_freq: int = 100,
    rank: int = 0,
):
    backbone.eval()
    head.eval()

    loss_meters = {"energy": AverageMeter(), "force": AverageMeter(), "total": AverageMeter()}
    mae_meters = {"energy": AverageMeter(), "force": AverageMeter()}
    mse_meters = {"energy": AverageMeter(), "force": AverageMeter()}

    train_cfg = config['training']
    t0 = time.perf_counter()

    for step, data in enumerate(data_loader):
        data = data.to(device)
        data = convert_to_escn_format(data)

        emb = backbone(data)
        outputs = head(data, emb)

        pred_energy = outputs['energy']['energy']
        pred_forces = outputs['forces']['forces']

        loss_e = energy_criterion(pred_energy, data.y)
        loss_f = force_criterion(pred_forces, data.force)
        loss = train_cfg['energy_weight'] * loss_e + train_cfg['force_weight'] * loss_f

        # Per-atom metrics
        natoms = data.natoms.float().view(-1)
        e_mae = torch.mean(torch.abs(data.y / natoms - pred_energy.detach() / natoms)).item()
        f_mae = torch.mean(torch.abs(data.force - pred_forces.detach())).item()
        e_mse = torch.mean((data.y / natoms - pred_energy.detach() / natoms) ** 2).item()
        f_mse = torch.mean((data.force - pred_forces.detach()) ** 2).item()

        bs = pred_energy.shape[0]
        loss_meters["energy"].update(loss_e.item(), n=bs)
        loss_meters["force"].update(loss_f.item(), n=pred_forces.shape[0])
        loss_meters["total"].update(loss.item(), n=bs)
        mae_meters["energy"].update(e_mae, n=bs)
        mae_meters["force"].update(f_mae, n=pred_forces.shape[0])
        mse_meters["energy"].update(e_mse, n=bs)
        mse_meters["force"].update(f_mse, n=pred_forces.shape[0])

        if rank == 0:
            _log_step_scalars(writer, split_name, {
                "loss_energy": loss_e.item(),
                "loss_force": loss_f.item(),
                "loss_total": loss.item(),
                "e_mae": e_mae,
                "f_mae": f_mae,
                "e_rmse": math.sqrt(e_mse),
                "f_rmse": math.sqrt(f_mse),
            }, global_step)
        global_step += 1

        if rank == 0 and ((step % print_freq == 0) or (step == len(data_loader) - 1)):
            elapsed = time.perf_counter() - t0
            speed_ms = 1e3 * elapsed / max(1, (step + 1))
            e_rmse = math.sqrt(mse_meters["energy"].avg)
            f_rmse = math.sqrt(mse_meters["force"].avg)
            print(
                f"[{split_name}] Step[{step}/{len(data_loader)}] "
                f"loss_e={loss_meters['energy'].avg:.5f} "
                f"loss_f={loss_meters['force'].avg:.5f} "
                f"loss_total={loss_meters['total'].avg:.5f} "
                f"e_MAE={mae_meters['energy'].avg:.5f} "
                f"f_MAE={mae_meters['force'].avg:.5f} "
                f"e_RMSE={e_rmse:.5f} "
                f"f_RMSE={f_rmse:.5f} "
                f"{speed_ms:.1f}ms/step"
            )

    if rank == 0:
        e_rmse = math.sqrt(mse_meters["energy"].avg)
        f_rmse = math.sqrt(mse_meters["force"].avg)
        _log_step_scalars(writer, split_name, {
            "epoch_loss_energy": loss_meters["energy"].avg,
            "epoch_loss_force": loss_meters["force"].avg,
            "epoch_loss_total": loss_meters["total"].avg,
            "epoch_e_mae": mae_meters["energy"].avg,
            "epoch_f_mae": mae_meters["force"].avg,
            "epoch_e_rmse": e_rmse,
            "epoch_f_rmse": f_rmse,
        }, 0)

    return mae_meters, mse_meters, loss_meters, global_step


def build_models(model_cfg, device):
    """Construct the same backbone and head for training and evaluation."""
    backbone = eSCNMDBackbone(
        max_num_elements=model_cfg.get('num_elements', 10),
        sphere_channels=model_cfg.get('sphere_channels', 128),
        lmax=model_cfg.get('lmax', 3),
        mmax=model_cfg.get('mmax', 2),
        cutoff=model_cfg.get('cutoff', 5.0),
        max_neighbors=model_cfg.get('max_neighbors', 50),
        num_layers=model_cfg.get('num_layers', 3),
        hidden_channels=model_cfg.get('hidden_channels', 128),
        edge_channels=model_cfg.get('edge_channels', 128),
        regress_forces=True,
        direct_forces=model_cfg.get('direct_forces', True),
        use_pbc=False,
        always_use_pbc=False,
        otf_graph=False,
        dataset_list=['dimer'],
        num_distance_basis=model_cfg.get('num_distance_basis', 512),
        ewald_hyperparams=model_cfg.get('ewald_hyperparams'),
        irreps=model_cfg.get('irreps', True),
    ).to(device)

    head = MLP_EFS_Head(backbone, wrap_property=True).to(device)

    return backbone, head


def train_worker(rank, world_size, config):
    """Worker function for each GPU process."""
    if world_size > 1:
        torch.cuda.set_device(rank)
        setup_distributed(rank, world_size, backend=config['system'].get('dist_backend', 'nccl'))
    elif config['system'].get('cuda', True) and torch.cuda.is_available():
        torch.cuda.set_device(0)

    data_cfg = config['data']
    train_cfg = config['training']
    model_cfg = config['model']
    output_cfg = config['output']
    system_cfg = config['system']

    ewald_flag = "ewald" if model_cfg.get('ewald_hyperparams') is not None else "no_ewald"

    timestamp = config.get('timestamp', time.strftime("%Y%m%d-%H%M%S"))
    run_name = f"dimer_escn_md_{ewald_flag}_{timestamp}"
    run_dir = os.path.join(output_cfg['base_dir'], run_name)

    ckpt_dir = os.path.join(run_dir, output_cfg['ckpt_dir'])
    log_dir = os.path.join(run_dir, output_cfg['log_dir'])

    if rank == 0:
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        dist.barrier()

    logger, log_file = setup_logging(log_dir, run_name, rank=rank)

    if rank == 0:
        logger.info("=" * 80)
        logger.info(f"Dimer Training with eSCN-MD (world_size={world_size})")
        logger.info("=" * 80)
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Ewald blocks: {ewald_flag}")
        logger.info(f"Loss type: {train_cfg.get('loss_type', 'mse')}")
        logger.info(f"GPUs: {system_cfg.get('gpus', list(range(world_size)))}")

    writer = SummaryWriter(log_dir=log_dir) if rank == 0 else None

    np.random.seed(data_cfg['seed'] + rank)
    torch.manual_seed(data_cfg['seed'] + rank)
    torch.cuda.manual_seed_all(data_cfg['seed'] + rank)

    if system_cfg['deterministic']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if system_cfg.get('cuda', True) and torch.cuda.is_available() else "cpu")
    if rank == 0:
        logger.info(f"Using device: {device}")

    # Load datasets
    if rank == 0:
        logger.info("Loading datasets...")

    train_set = DimerDataset(data_cfg['root'], data_cfg['train_xyz'], split="train")
    val_set = DimerDataset(data_cfg['root'], data_cfg['val_xyz'], split="val")

    if rank == 0:
        logger.info(f"Train set: {len(train_set)} samples")
        logger.info(f"Val set:   {len(val_set)} samples")

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=system_cfg['num_workers'],
        persistent_workers=True if system_cfg['num_workers'] > 0 else False,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=train_cfg['eval_batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=system_cfg['num_workers'],
        persistent_workers=True if system_cfg['num_workers'] > 0 else False,
        pin_memory=True
    )

    # Create model
    if rank == 0:
        logger.info("Creating eSCN-MD model...")

    backbone, head = build_models(model_cfg, device)

    if world_size > 1:
        backbone = DDP(
            backbone,
            device_ids=[rank],
            find_unused_parameters=system_cfg.get('find_unused_parameters', False)
        )
        head = DDP(
            head,
            device_ids=[rank],
            find_unused_parameters=system_cfg.get('find_unused_parameters', False)
        )

    if rank == 0:
        logger.info("Model parameters:")
        logger.info("Backbone:")
        count_params(backbone, logger=logger)
        logger.info("Head:")
        count_params(head, logger=logger)

        total_params = sum(p.numel() for p in (backbone.module if world_size > 1 else backbone).parameters())
        total_params += sum(p.numel() for p in (head.module if world_size > 1 else head).parameters())
        logger.info(f"Total parameters: {total_params / 1e6:.3f}M")

    # Loss functions
    loss_type = train_cfg.get('loss_type', 'mse')
    if loss_type == 'mse':
        energy_criterion = nn.MSELoss()
        force_criterion = nn.MSELoss()
    else:
        energy_criterion = nn.L1Loss()
        force_criterion = nn.L1Loss()

    if rank == 0:
        logger.info(f"Loss type: {loss_type}")

    # Optimizer
    all_params = list(backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(
        all_params,
        lr=train_cfg['lr'],
        weight_decay=train_cfg.get('weight_decay', 0.0),
        amsgrad=train_cfg.get('amsgrad', True),
    )

    # LR schedulers
    lr_warmup_steps = int(train_cfg.get('lr_warmup_steps', 0))
    warmup_start_factor = float(train_cfg.get('lr_warmup_start_factor', 1e-3))
    if lr_warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=lr_warmup_steps,
        )
        if rank == 0:
            logger.info(f"LR warmup: {lr_warmup_steps} steps, start_factor={warmup_start_factor}")
    else:
        warmup_scheduler = None

    scheduler_name = str(train_cfg.get('lr_scheduler', 'step')).lower()
    if scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_cfg.get('lr_step_size', 15),
            gamma=train_cfg.get('lr_gamma', 0.9),
        )
    elif scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg['epochs'],
            eta_min=train_cfg.get('lr_min', 1e-6),
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler='{scheduler_name}'. Use 'step' or 'cosine'.")

    if rank == 0:
        logger.info(f"LR scheduler: {scheduler_name}")

    global_steps = {"train": 0, "val": 0}

    if rank == 0:
        logger.info("=" * 80)
        logger.info("Starting training...")
        logger.info("=" * 80)

    best_val_loss = float("inf")
    best_epoch = 0
    best_val_mae = {"energy": 0.0, "force": 0.0}
    # Top-K checkpoint tracking: list of (val_loss, epoch, path)
    top_k = train_cfg.get('top_k_checkpoints', 3)
    best_checkpoints = []

    for epoch in range(1, train_cfg['epochs'] + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        tr_mae, tr_mse, tr_loss, global_steps["train"] = train_one_epoch(
            config, backbone, head, energy_criterion, force_criterion,
            train_loader, optimizer, device, epoch,
            writer=writer, global_step=global_steps["train"],
            print_freq=train_cfg['print_freq'],
            rank=rank, world_size=world_size,
            warmup_scheduler=warmup_scheduler,
            lr_warmup_steps=lr_warmup_steps,
        )

        # Disable warmup after it completes
        if warmup_scheduler is not None and global_steps["train"] >= lr_warmup_steps:
            if rank == 0:
                logger.info(f"Warmup completed at step {global_steps['train']}")
            warmup_scheduler = None

        va_mae, va_mse, va_loss, global_steps["val"] = evaluate(
            config, backbone, head, energy_criterion, force_criterion,
            val_loader, device, writer=writer, split_name="val",
            global_step=global_steps["val"], print_freq=train_cfg['print_freq'],
            rank=rank
        )

        val_loss_total = va_loss["total"].avg

        # Step epoch-level scheduler only after warmup is done
        if warmup_scheduler is None:
            scheduler.step()

        if rank == 0:
            tr_e_rmse = math.sqrt(tr_mse["energy"].avg)
            tr_f_rmse = math.sqrt(tr_mse["force"].avg)
            va_e_rmse = math.sqrt(va_mse["energy"].avg)
            va_f_rmse = math.sqrt(va_mse["force"].avg)

            writer.add_scalars("epoch_summary/eMAE", {
                "train_energy": tr_mae["energy"].avg,
                "train_force": tr_mae["force"].avg,
                "val_energy": va_mae["energy"].avg,
                "val_force": va_mae["force"].avg,
            }, epoch)
            writer.add_scalars("epoch_summary/eRMSE", {
                "train_energy": tr_e_rmse,
                "train_force": tr_f_rmse,
                "val_energy": va_e_rmse,
                "val_force": va_f_rmse,
            }, epoch)
            writer.add_scalars("epoch_summary/loss", {
                "train_energy": tr_loss["energy"].avg,
                "train_force": tr_loss["force"].avg,
                "train_total": tr_loss["total"].avg,
                "val_energy": va_loss["energy"].avg,
                "val_force": va_loss["force"].avg,
                "val_total": va_loss["total"].avg,
            }, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("train/learning_rate_epoch", current_lr, epoch)

            logger.info(
                f"Epoch {epoch:03d} | "
                f"train: loss={tr_loss['total'].avg:.6f} "
                f"eMAE={tr_mae['energy'].avg:.6f} fMAE={tr_mae['force'].avg:.6f} "
                f"eRMSE={tr_e_rmse:.6f} fRMSE={tr_f_rmse:.6f} | "
                f"val: loss={va_loss['total'].avg:.6f} "
                f"eMAE={va_mae['energy'].avg:.6f} fMAE={va_mae['force'].avg:.6f} "
                f"eRMSE={va_e_rmse:.6f} fRMSE={va_f_rmse:.6f} | "
                f"lr={current_lr:.2e}"
            )

            # Checkpoint saving
            backbone_state = backbone.module.state_dict() if world_size > 1 else backbone.state_dict()
            head_state = head.module.state_dict() if world_size > 1 else head.state_dict()

            ckpt_data = {
                "backbone_state_dict": backbone_state,
                "head_state_dict": head_state,
                "config": config,
                "epoch": epoch,
                "val_loss": val_loss_total,
                "val_mae_energy": va_mae["energy"].avg,
                "val_mae_force": va_mae["force"].avg,
            }

            # Top-K checkpoint management
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch}_valloss{val_loss_total:.6f}.pt")
            best_checkpoints.append((val_loss_total, epoch, ckpt_path))
            best_checkpoints.sort(key=lambda x: x[0])

            if len(best_checkpoints) > top_k:
                _, _, path_to_remove = best_checkpoints.pop()
                if os.path.exists(path_to_remove):
                    os.remove(path_to_remove)

            # Save if in top K
            if any(p == ckpt_path for _, _, p in best_checkpoints):
                torch.save(ckpt_data, ckpt_path)
                logger.info(f"  -> Saved top-{top_k} checkpoint: {os.path.basename(ckpt_path)}")

            # Save best
            if val_loss_total < best_val_loss:
                best_val_loss = val_loss_total
                best_epoch = epoch
                best_val_mae = {
                    "energy": va_mae["energy"].avg,
                    "force": va_mae["force"].avg,
                }
                ckpt_data["best_val_loss"] = best_val_loss
                torch.save(ckpt_data, os.path.join(ckpt_dir, "best_dimer.pt"))
                logger.info(f"  -> New best model (val_loss: {val_loss_total:.6f})")

            # Save last
            if output_cfg['save_last']:
                torch.save(ckpt_data, os.path.join(ckpt_dir, "last_dimer.pt"))

    # Final summary
    if rank == 0:
        logger.info("")
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Best model: epoch {best_epoch}, val_loss={best_val_loss:.6f}")
        logger.info(f"  eMAE={best_val_mae['energy']:.6f}, fMAE={best_val_mae['force']:.6f}")
        logger.info(f"\nTop-{top_k} checkpoints by val loss:")
        for i, (loss, ep, path) in enumerate(best_checkpoints, 1):
            logger.info(f"  {i}. Epoch {ep}, val_loss={loss:.6f} - {os.path.basename(path)}")
        logger.info("=" * 80)

        summary_path = os.path.join(run_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRAINING SUMMARY (Dimer eSCN-MD)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Dataset: dimer\n")
            f.write(f"Ewald blocks: {ewald_flag}\n")
            f.write(f"Loss type: {loss_type}\n")
            f.write(f"LR scheduler: {scheduler_name}\n")
            f.write(f"Total epochs: {train_cfg['epochs']}\n")
            f.write(f"Best Model (Epoch {best_epoch}):\n")
            f.write(f"  Val Loss:  {best_val_loss:.6f}\n")
            f.write(f"  Val eMAE:  {best_val_mae['energy']:.6f}\n")
            f.write(f"  Val fMAE:  {best_val_mae['force']:.6f}\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Results saved to: {run_dir}")

        writer.flush()
        writer.close()

    if world_size > 1:
        cleanup_distributed()


def main():
    args = get_args()

    if args.mode == "test" and args.checkpoint is None:
        print("ERROR: --checkpoint is required in test mode")
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"ERROR: Config file {args.config} not found")
        sys.exit(1)
    config = load_config(args.config)

    config = merge_args_with_config(args, config)

    system_cfg = config['system']
    gpu_ids = system_cfg.get('gpus', [])
    if len(gpu_ids) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
        print(f"Setting CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}")

    config['timestamp'] = time.strftime("%Y%m%d-%H%M%S")

    if args.mode == "test":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        # Architecture and normalization must come from the saved checkpoint.
        config = merge_args_with_config(args, checkpoint['config'])
        device = torch.device("cuda" if config['system'].get('cuda', True) and torch.cuda.is_available() else "cpu")
        backbone, head = build_models(config['model'], device)
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
        head.load_state_dict(checkpoint['head_state_dict'])
        data_cfg = config['data']
        dataset = DimerDataset(data_cfg['root'], data_cfg['val_xyz'], split="test")
        loader = DataLoader(dataset, batch_size=config['training']['eval_batch_size'], shuffle=False)
        criterion = nn.MSELoss() if config['training'].get('loss_type', 'mse') == 'mse' else nn.L1Loss()
        mae, mse, losses, _ = evaluate(config, backbone, head, criterion, criterion,
                                       loader, device, split_name="evaluation")
        print(f"Evaluation samples: {len(dataset)}; energy metrics are per atom")
        print(f"energy_MAE={mae['energy'].avg:.8f} force_MAE={mae['force'].avg:.8f} "
              f"energy_RMSE={math.sqrt(mse['energy'].avg):.8f} "
              f"force_RMSE={math.sqrt(mse['force'].avg):.8f} loss={losses['total'].avg:.8f}")
    else:
        data_cfg = config['data']
        train_cfg = config['training']

        print(f"\n{'=' * 80}")
        print("Running in TRAIN mode")
        print(f"  Config file: {args.config}")
        print(f"  Dataset: dimer")
        print(f"  Loss type: {train_cfg.get('loss_type', 'mse')}")
        print(f"  LR scheduler: {train_cfg.get('lr_scheduler', 'step')}")
        print(f"  Epochs: {train_cfg['epochs']}")
        print(f"  GPUs: {system_cfg.get('gpus', [0])}")
        print(f"{'=' * 80}\n")

        if system_cfg.get('distributed', False) and len(system_cfg.get('gpus', [])) > 1:
            world_size = len(system_cfg['gpus'])
            print(f"Starting distributed training on {world_size} GPUs: {system_cfg['gpus']}")
            mp.spawn(
                train_worker,
                args=(world_size, config),
                nprocs=world_size,
                join=True
            )
        else:
            if system_cfg.get('cuda', True) and torch.cuda.is_available():
                if len(system_cfg.get('gpus', [])) > 0:
                    device_id = system_cfg['gpus'][0]
                    print(f"Starting single GPU training on GPU {device_id}")
                    # CUDA_VISIBLE_DEVICES remaps the chosen physical GPU to 0.
                    torch.cuda.set_device(0)
            train_worker(0, 1, config)


if __name__ == "__main__":
    main()
