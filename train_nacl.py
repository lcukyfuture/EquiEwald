#!/usr/bin/env python3
"""Train a standalone periodic eSCN-irreps + Ewald model on Molten-NaCl.

This script intentionally does not import the existing non-periodic
``escn_md_irreps`` backbone/head or the existing irreps EwaldBlock. It keeps the
PBC-specific backbone, Ewald block, and EFS head local to this file while
reusing low-level eSCN/OC20 utilities.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR
DEFAULT_XYZ = REPO_DIR / "data" / "nacl" / "raw" / "ML_AB_NaCl_liquid_Extract.xyz"
DEFAULT_OUT = REPO_DIR / "runs" / "nacl_escn_irreps_pbc"
JD_PATH = REPO_DIR / "fairchem" / "core" / "models" / "uma" / "Jd.pt"


def configure_paths() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "equiewald-matplotlib-cache"))
    for path in (str(REPO_DIR), str(THIS_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


configure_paths()

from ase.io import read  # noqa: E402
from fairchem.core.common import gp_utils  # noqa: E402
from fairchem.core.common.distutils import get_device_for_local_rank  # noqa: E402
from fairchem.core.common.utils import conditional_grad  # noqa: E402
from fairchem.core.graph.compute import generate_graph  # noqa: E402
from fairchem.core.models.uma.common.rotation import init_edge_rot_mat, rotation_to_wigner  # noqa: E402
from fairchem.core.models.uma.common.rotation_cuda_graph import RotMatWignerCudaGraph  # noqa: E402
from fairchem.core.models.uma.common.so3 import CoefficientMapping, SO3_Grid  # noqa: E402
from fairchem.core.models.uma.escn_md_block import eSCNMD_Block  # noqa: E402
from fairchem.core.models.uma.nn.embedding_dev import ChgSpinEmbedding, EdgeDegreeEmbedding  # noqa: E402
from fairchem.core.models.uma.nn.layer_norm import (  # noqa: E402
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from fairchem.core.models.uma.nn.radial import GaussianSmearing  # noqa: E402
from fairchem.core.models.uma.nn.so3_layers import SO3_Linear  # noqa: E402
from ocpmodels.equiformer_v2.activation import GateActivation  # noqa: E402
from ocpmodels.equiformer_v2.so3 import SO3_Embedding, SO3_LinearV2  # noqa: E402
from ocpmodels.models.gemnet.layers.base_layers import Dense, ResidualLayer  # noqa: E402
from torch_geometric.data import Batch, Data  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402


ESCNMD_DEFAULT_EDGE_CHUNK_SIZE = 1024 * 128


def get_k_index_product_set(num_k_x: int, num_k_y: int, num_k_z: int) -> tuple[torch.Tensor, int]:
    k_index_sets = (
        torch.arange(-num_k_x, num_k_x + 1, dtype=torch.float),
        torch.arange(-num_k_y, num_k_y + 1, dtype=torch.float),
        torch.arange(-num_k_z, num_k_z + 1, dtype=torch.float),
    )
    k_index_product_set = torch.cartesian_prod(*k_index_sets)
    k_index_product_set = k_index_product_set[k_index_product_set.shape[0] // 2 + 1 :]
    return k_index_product_set, int(k_index_product_set.shape[0])


def x_to_k_cell(cells: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cross_a2a3 = torch.cross(cells[:, 1], cells[:, 2], dim=-1)
    cross_a3a1 = torch.cross(cells[:, 2], cells[:, 0], dim=-1)
    cross_a1a2 = torch.cross(cells[:, 0], cells[:, 1], dim=-1)
    vol = torch.sum(cells[:, 0] * cross_a2a3, dim=-1, keepdim=True)
    b1 = 2 * math.pi * cross_a2a3 / vol
    b2 = 2 * math.pi * cross_a3a1 / vol
    b3 = 2 * math.pi * cross_a1a2 / vol
    return torch.stack((b1, b2, b3), dim=1), vol[:, 0]


def default_config() -> dict[str, Any]:
    return {
        "seed": 7,
        "data_path": str(DEFAULT_XYZ),
        "output_dir": str(DEFAULT_OUT),
        "unique_output_dir": True,
        "run_name": "escn_irreps_molten_nacl_pbc",
        "valid_fraction": 0.1,
        "train_fraction": None,
        "cutoff": 5.29,
        "batch_size": 1,
        "num_epochs": 5,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "amsgrad": True,
        "lr_step_size": 15,
        "lr_gamma": 0.9,
        "energy_weight": 1.0,
        "force_weight": 100.0,
        "grad_clip_norm": 10.0,
        "patience": 20,
        "min_delta": 1e-4,
        "max_frames": None,
        "smoke": False,
        "resume": None,
        "save_best_history": False,
        "model": {
            "sphere_channels": 128,
            "lmax": 3,
            "mmax": 2,
            "max_neighbors": 80,
            "num_layers": 3,
            "hidden_channels": 128,
            "edge_channels": 128,
            "num_distance_basis": 512,
            "num_k_x": 1,
            "num_k_y": 1,
            "num_k_z": 3,
            "downprojection": 8,
            "num_hidden": 1,
        },
    }


def merge_config(base: dict[str, Any], updates: dict[str, Any], *, path: str = "config") -> dict[str, Any]:
    for key, value in updates.items():
        if key not in base:
            raise KeyError(f"Unknown {path} key: {key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f"{path}.{key} must be an object")
            merge_config(base[key], value, path=f"{path}.{key}")
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"YAML config must contain a mapping at top level: {config_path}")
    return loaded


def sanitize_run_name(run_name: str | None) -> str:
    raw_name = (run_name or "run").strip()
    safe_name = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in raw_name)
    return safe_name.strip("._-") or "run"


def resolve_output_dir(config: dict[str, Any]) -> Path:
    output_root = Path(config["output_dir"]).expanduser()
    if not bool(config["unique_output_dir"]):
        return output_root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = sanitize_run_name(config.get("run_name"))
    return output_root / f"{run_name}_{timestamp}_pid{os.getpid()}"


def write_effective_config(output_dir: Path, config: dict[str, Any]) -> None:
    config_path = output_dir / "effective_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def setup_logging(output_dir: Path) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger = logging.getLogger("escn_irreps_molten_nacl_pbc")
    logger.info("Logging to %s", log_path)
    return logger


def atoms_to_data(atoms) -> Data:
    pos = torch.as_tensor(atoms.get_positions(), dtype=torch.get_default_dtype())
    atomic_numbers = torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    if "forces" in atoms.arrays:
        forces_array = atoms.arrays["forces"]
    else:
        forces_array = atoms.get_forces()
    if "energy" in atoms.info:
        energy_value = atoms.info["energy"]
    elif atoms.calc is not None and "energy" in atoms.calc.results:
        energy_value = atoms.calc.results["energy"]
    else:
        energy_value = atoms.get_potential_energy()
    forces = torch.as_tensor(forces_array, dtype=pos.dtype)
    energy = torch.tensor([[float(energy_value)]], dtype=pos.dtype)
    cell = torch.as_tensor(atoms.cell.array, dtype=pos.dtype).view(1, 3, 3)
    pbc = torch.as_tensor(atoms.pbc, dtype=torch.bool).view(1, 3)
    if not bool(pbc.all()):
        raise ValueError(f"Expected full 3D PBC for Molten-NaCl frame, got {atoms.pbc}")
    return Data(
        pos=pos,
        positions=pos,
        atomic_numbers=atomic_numbers,
        batch=torch.zeros(pos.shape[0], dtype=torch.long),
        natoms=torch.tensor([pos.shape[0]], dtype=torch.long),
        cell=cell,
        pbc=pbc,
        charge=torch.zeros(1, dtype=pos.dtype),
        spin=torch.zeros(1, dtype=pos.dtype),
        y=energy,
        force=forces,
    )


def load_dataset(path: str | Path) -> list[Data]:
    atoms_list = read(str(path), ":")
    if not atoms_list:
        raise RuntimeError(f"No structures found in {path}")
    return [atoms_to_data(atoms) for atoms in atoms_list]


def split_dataset(dataset: list[Data], valid_fraction: float, seed: int) -> tuple[list[Data], list[Data]]:
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be between 0 and 1")
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_valid = max(1, int(round(len(indices) * valid_fraction)))
    n_valid = min(n_valid, len(indices) - 1)
    valid_idx = set(indices[:n_valid])
    train = [dataset[i] for i in indices if i not in valid_idx]
    valid = [dataset[i] for i in indices if i in valid_idx]
    return train, valid


class PBCIrrepsEwaldBlock(nn.Module):
    """PBC-only irreps Ewald message block."""

    def __init__(
        self,
        shared_downprojection: Dense,
        emb_size_atom: int,
        downprojection_size: int,
        num_hidden: int,
        lmax_list: list[int],
        mmax_list: list[int],
        channel: int,
        activation: str | None = None,
        return_k_params: bool = True,
    ):
        super().__init__()
        self.return_k_params = return_k_params
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.channel = channel
        self.l_max = lmax_list[0]
        self.down = shared_downprojection
        self.up = Dense(downprojection_size, emb_size_atom, activation=None, bias=False)
        self.residual = SO3_LinearV2(self.channel, self.channel, lmax=self.l_max)
        self.ewald_layers = nn.ModuleList(
            [self.get_mlp(emb_size_atom, emb_size_atom, num_hidden, activation) for _ in range(self.l_max + 1)]
        )
        self.gate0_linear_1 = nn.Linear(self.channel, max(self.lmax_list) * self.channel)
        self.gate0_linear_2 = nn.Linear(self.channel, max(self.lmax_list) * self.channel)
        self.gate_act_1 = GateActivation(
            lmax=max(self.lmax_list),
            mmax=max(self.mmax_list),
            num_channels=self.channel,
        )
        self.gate_act_2 = GateActivation(
            lmax=max(self.lmax_list),
            mmax=max(self.mmax_list),
            num_channels=self.channel,
        )

    @staticmethod
    def get_mlp(units_in: int, units: int, num_hidden: int, activation: str | None):
        mlp = [Dense(units_in, units, activation=activation, bias=False)]
        mlp += [ResidualLayer(units, nLayers=2, activation=activation) for _ in range(num_hidden)]
        return nn.ModuleList(mlp)

    def forward(
        self,
        h: SO3_Embedding,
        x: torch.Tensor,
        k: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        dot: torch.Tensor | None = None,
        sinc_damping: torch.Tensor | None = None,
    ):
        res_embedding = self.residual(h)
        x_0_gating = self.gate0_linear_1(res_embedding.embedding[:, 0, :])
        res_embedding.embedding = self.gate_act_1(x_0_gating, res_embedding.embedding)
        res_embed = res_embedding.embedding

        if dot is None:
            b = batch_seg.view(-1, 1, 1).expand(-1, k.shape[-2], k.shape[-1])
            dot = torch.sum(torch.gather(k, 0, b) * x.unsqueeze(-2), dim=-1)

        if sinc_damping is None:
            sinc_damping = torch.ones_like(dot)

        cos_dot = torch.cos(dot).unsqueeze(-1).unsqueeze(-1)
        sin_dot = torch.sin(dot).unsqueeze(-1).unsqueeze(-1)
        damping = sinc_damping.unsqueeze(-1).unsqueeze(-1)
        base_filter = torch.matmul(self.up.linear.weight, self.down.linear.weight).T
        k_filter_pbc = base_filter.unsqueeze(0).expand(num_batch, -1, -1).unsqueeze(-2)

        # ---- FUSED: process all SO(3) degrees in a single scatter to reduce
        #        GPU kernel launch overhead (was 8 index_add_ calls, now 2) ----
        total_m = res_embed.shape[1]           # 16 = Σ(2l+1) for lmax=3
        h_exp = res_embed.unsqueeze(1)         # [n_atoms, 1, total_m, C]

        sf_real = torch.zeros(
            num_batch, dot.shape[1], total_m, res_embed.shape[-1],
            device=res_embed.device, dtype=res_embed.dtype,
        ).index_add_(0, batch_seg, h_exp * cos_dot * damping)
        sf_imag = torch.zeros_like(sf_real).index_add_(
            0, batch_seg, h_exp * sin_dot * damping)

        sf_real = sf_real * k_filter_pbc
        sf_imag = sf_imag * k_filter_pbc

        real_part = torch.index_select(sf_real, 0, batch_seg)
        imag_part = torch.index_select(sf_imag, 0, batch_seg)
        h_update = 0.01 * torch.sum(
            (real_part * cos_dot + imag_part * sin_dot) * damping, dim=1)
        # h_update: [n_atoms, total_m, C]

        # ---- Per-degree MLPs (lightweight, O(n_atoms) per degree) ----
        h_update_list = []
        offset = 0
        for degree in range(self.l_max + 1):
            num_m = 2 * degree + 1
            h_update_l = h_update[:, offset:offset + num_m, :]
            offset += num_m
            for layer in self.ewald_layers[degree]:
                h_update_l = layer(h_update_l)
            h_update_list.append(h_update_l)

        h_so3 = SO3_Embedding(
            0,
            lmax_list=self.lmax_list,
            num_channels=self.channel,
            device=res_embed.device,
            dtype=res_embed.dtype,
        )
        h_so3.set_embedding(torch.cat(h_update_list, dim=1))
        h_so3.set_lmax_mmax(self.lmax_list, self.lmax_list)
        x_0_gating_2 = self.gate0_linear_2(h_so3.embedding[:, 0, :])
        h_output = self.gate_act_2(x_0_gating_2, h_so3.embedding)
        if self.return_k_params:
            return h_output, dot, sinc_damping
        return h_output


class PBCeSCNIrrepsBackbone(nn.Module):
    """eSCN-irreps backbone with only the true-PBC graph/Ewald path enabled."""

    def __init__(
        self,
        max_num_elements: int,
        sphere_channels: int = 128,
        lmax: int = 3,
        mmax: int = 2,
        grid_resolution: int | None = None,
        max_neighbors: int = 80,
        cutoff: float = 5.29,
        edge_channels: int = 128,
        num_distance_basis: int = 512,
        num_layers: int = 3,
        hidden_channels: int = 128,
        norm_type: str = "rms_norm_sh",
        act_type: str = "gate",
        ff_type: str = "grid",
        chg_spin_emb_type: str = "pos_emb",
        radius_pbc_version: int = 1,
        num_k_x: int = 2,
        num_k_y: int = 2,
        num_k_z: int = 2,
        downprojection: int = 8,
        num_hidden: int = 1,
    ):
        super().__init__()
        self.max_num_elements = max_num_elements
        self.lmax = lmax
        self.mmax = mmax
        self.sphere_channels = sphere_channels
        self.grid_resolution = grid_resolution
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.radius_pbc_version = radius_pbc_version
        self.enforce_max_neighbors_strictly = False
        self.regress_forces = True
        self.direct_forces = False
        self.regress_stress = False
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.use_cuda_graph_wigner = False
        self.rot_mat_wigner_cuda = None

        if not JD_PATH.is_file():
            raise FileNotFoundError(f"Missing Wigner Jd table: {JD_PATH}")
        try:
            jd_list = torch.load(JD_PATH, weights_only=True)
        except TypeError:
            jd_list = torch.load(JD_PATH)
        for degree in range(self.lmax + 1):
            self.register_buffer(f"Jd_{degree}", jd_list[degree])
        self.sph_feature_size = int((self.lmax + 1) ** 2)
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)

        self.SO3_grid = nn.ModuleDict(
            {
                "lmax_lmax": SO3_Grid(self.lmax, self.lmax, resolution=grid_resolution, rescale=True),
                "lmax_mmax": SO3_Grid(self.lmax, self.mmax, resolution=grid_resolution, rescale=True),
            }
        )

        self.sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels)
        self.charge_embedding = ChgSpinEmbedding(chg_spin_emb_type, "charge", self.sphere_channels, grad=False)
        self.spin_embedding = ChgSpinEmbedding(chg_spin_emb_type, "spin", self.sphere_channels, grad=False)
        self.mix_csd = nn.Linear(2 * self.sphere_channels, self.sphere_channels)

        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, num_distance_basis, 2.0)
        self.source_embedding = nn.Embedding(self.max_num_elements, edge_channels)
        self.target_embedding = nn.Embedding(self.max_num_elements, edge_channels)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
        self.edge_channels_list = [num_distance_basis + 2 * edge_channels, edge_channels, edge_channels]
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            sphere_channels=self.sphere_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            max_num_elements=self.max_num_elements,
            edge_channels_list=self.edge_channels_list,
            rescale_factor=5.0,
            cutoff=self.cutoff,
            mappingReduced=self.mappingReduced,
            activation_checkpoint_chunk_size=None,
        )

        self.blocks = nn.ModuleList(
            [
                eSCNMD_Block(
                    self.sphere_channels,
                    self.hidden_channels,
                    self.lmax,
                    self.mmax,
                    self.mappingReduced,
                    self.SO3_grid,
                    self.edge_channels_list,
                    self.cutoff,
                    norm_type,
                    act_type,
                    ff_type,
                    activation_checkpoint_chunk_size=None,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm = get_normalization_layer(norm_type, lmax=self.lmax, num_channels=self.sphere_channels)
        coefficient_index = self.SO3_grid["lmax_lmax"].mapping.coefficient_idx(self.lmax, self.mmax)
        self.register_buffer("coefficient_index", coefficient_index, persistent=False)

        self.k_index_product_set, self.num_k_degrees_of_freedom = get_k_index_product_set(
            num_k_x,
            num_k_y,
            num_k_z,
        )
        self.down = Dense(self.num_k_degrees_of_freedom, downprojection, activation=None, bias=False)
        self.ewald_blocks = nn.ModuleList(
            [
                PBCIrrepsEwaldBlock(
                    self.down,
                    self.sphere_channels,
                    downprojection,
                    num_hidden,
                    [self.lmax],
                    [self.mmax],
                    self.sphere_channels,
                    activation="silu",
                )
                for _ in range(self.num_layers)
            ]
        )

    def csd_embedding(self, charge: torch.Tensor, spin: torch.Tensor) -> torch.Tensor:
        chg_emb = self.charge_embedding(charge)
        spin_emb = self.spin_embedding(spin)
        return torch.nn.SiLU()(self.mix_csd(torch.cat((chg_emb, spin_emb), dim=1)))

    def _get_rotmat_and_wigner(self, edge_distance_vecs: torch.Tensor, use_cuda_graph: bool):
        jd_buffers = [getattr(self, f"Jd_{degree}").type(edge_distance_vecs.dtype) for degree in range(self.lmax + 1)]
        if use_cuda_graph:
            if self.rot_mat_wigner_cuda is None:
                self.rot_mat_wigner_cuda = RotMatWignerCudaGraph()
            edge_rot_mat, wigner, wigner_inv = self.rot_mat_wigner_cuda.get_rotmat_and_wigner(
                edge_distance_vecs,
                jd_buffers,
            )
        else:
            edge_rot_mat = init_edge_rot_mat(edge_distance_vecs, rot_clip=True)
            wigner = rotation_to_wigner(edge_rot_mat, 0, self.lmax, jd_buffers, rot_clip=True)
            wigner_inv = torch.transpose(wigner, 1, 2).contiguous()

        if self.mmax != self.lmax:
            wigner = wigner.index_select(1, self.coefficient_index)
            wigner_inv = wigner_inv.index_select(2, self.coefficient_index)

        wigner_and_m_mapping = torch.einsum("mk,nkj->nmj", self.mappingReduced.to_m, wigner)
        wigner_and_m_mapping_inv = torch.einsum("njk,mk->njm", wigner_inv, self.mappingReduced.to_m)
        return edge_rot_mat, wigner_and_m_mapping, wigner_and_m_mapping_inv

    def _generate_graph(self, data: Batch) -> dict[str, torch.Tensor]:
        if not hasattr(data, "pbc") or not bool(data.pbc.all()):
            raise ValueError("PBCeSCNIrrepsBackbone requires all structures to have pbc=[True, True, True]")
        graph_dict = generate_graph(
            data,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
            radius_pbc_version=self.radius_pbc_version,
            pbc=data.pbc,
        )
        graph_dict["node_offset"] = 0
        graph_dict["edge_distance_vec_full"] = graph_dict["edge_distance_vec"]
        graph_dict["edge_distance_full"] = graph_dict["edge_distance"]
        graph_dict["edge_index_full"] = graph_dict["edge_index"]
        return graph_dict

    @conditional_grad(torch.enable_grad())
    def forward(self, data: Batch) -> dict[str, torch.Tensor]:
        data.atomic_numbers = data.atomic_numbers.long()
        if data.pos.requires_grad is False:
            data.pos.requires_grad = True

        batch_size = int(data.natoms.numel())
        csd_mixed_emb = self.csd_embedding(data.charge, data.spin)
        graph_dict = self._generate_graph(data)
        if graph_dict["edge_index"].numel() == 0:
            raise ValueError(f"No PBC edges found with cutoff={self.cutoff}")

        _, wigner_and_m_mapping, wigner_and_m_mapping_inv = self._get_rotmat_and_wigner(
            graph_dict["edge_distance_vec_full"],
            use_cuda_graph=self.use_cuda_graph_wigner and "cuda" in get_device_for_local_rank() and not self.training,
        )

        x_message = torch.zeros(
            data.atomic_numbers.shape[0],
            self.sph_feature_size,
            self.sphere_channels,
            device=data.pos.device,
            dtype=data.pos.dtype,
        )
        x_message[:, 0, :] = self.sphere_embedding(data.atomic_numbers)
        sys_node_embedding = csd_mixed_emb[data.batch]
        x_message[:, 0, :] = x_message[:, 0, :] + sys_node_embedding

        edge_distance_embedding = self.distance_expansion(graph_dict["edge_distance"])
        source_embedding = self.source_embedding(data.atomic_numbers[graph_dict["edge_index"][0]])
        target_embedding = self.target_embedding(data.atomic_numbers[graph_dict["edge_index"][1]])
        x_edge = torch.cat((edge_distance_embedding, source_embedding, target_embedding), dim=1)
        x_message = self.edge_degree_embedding(
            x_message,
            x_edge,
            graph_dict["edge_distance"],
            graph_dict["edge_index"],
            wigner_and_m_mapping_inv,
            graph_dict["node_offset"],
        )

        k_cell, _ = x_to_k_cell(data.cell)
        k_grid = torch.matmul(self.k_index_product_set.to(device=data.pos.device, dtype=data.pos.dtype), k_cell)
        dot = None
        sinc_damping = None
        for layer_idx in range(self.num_layers):
            x_so3 = SO3_Embedding(
                0,
                lmax_list=[self.lmax],
                num_channels=self.sphere_channels,
                device=x_message.device,
                dtype=x_message.dtype,
            )
            x_so3.set_embedding(x_message)
            x_so3.set_lmax_mmax([self.lmax], [self.lmax])
            dx_ewald, dot, sinc_damping = self.ewald_blocks[layer_idx](
                x_so3,
                data.pos,
                k_grid,
                batch_size,
                data.batch,
                dot,
                sinc_damping,
            )
            dx_message = self.blocks[layer_idx](
                x_message,
                x_edge,
                graph_dict["edge_distance"],
                graph_dict["edge_index"],
                wigner_and_m_mapping,
                wigner_and_m_mapping_inv,
                sys_node_embedding=sys_node_embedding,
                node_offset=graph_dict["node_offset"],
            )
            x_message = (x_message + dx_ewald + dx_message) / math.sqrt(3.0)

        return {
            "node_embedding": self.norm(x_message),
            "batch": data.batch,
            "displacement": None,
            "orig_cell": None,
        }

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(
                module,
                (
                    nn.Linear,
                    SO3_Linear,
                    nn.LayerNorm,
                    EquivariantLayerNormArray,
                    EquivariantLayerNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonics,
                    EquivariantRMSNormArraySphericalHarmonicsV2,
                ),
            ):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, (nn.Linear, SO3_Linear)) and "weight" in parameter_name:
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    if global_parameter_name in named_parameters_list:
                        no_wd_list.append(global_parameter_name)
        return set(no_wd_list)


class PBCEFSHead(nn.Module):
    """Energy-conserving scalar energy and force head."""

    def __init__(self, backbone: PBCeSCNIrrepsBackbone):
        super().__init__()
        self.backbone = backbone
        self.regress_forces = True
        self.direct_forces = False
        self.regress_stress = False
        self.energy_block = nn.Sequential(
            nn.Linear(backbone.sphere_channels, backbone.hidden_channels),
            nn.SiLU(),
            nn.Linear(backbone.hidden_channels, backbone.hidden_channels),
            nn.SiLU(),
            nn.Linear(backbone.hidden_channels, 1),
        )

    @staticmethod
    def _batch_sum(node_values: torch.Tensor, batch: torch.Tensor, batch_size: int) -> torch.Tensor:
        graph_values = torch.zeros(batch_size, 1, device=node_values.device, dtype=node_values.dtype)
        graph_values.index_add_(0, batch, node_values.view(-1, 1))
        return graph_values

    @conditional_grad(torch.enable_grad())
    def forward(self, data: Batch) -> dict[str, torch.Tensor]:
        emb = self.backbone(data)
        node_scalar = emb["node_embedding"].narrow(1, 0, 1).squeeze(1)
        node_energy = self.energy_block(node_scalar)
        energy_part = self._batch_sum(node_energy, data.batch, int(data.natoms.numel()))
        energy = gp_utils.reduce_from_model_parallel_region(energy_part) if gp_utils.initialized() else energy_part
        forces = -torch.autograd.grad(
            energy.sum(),
            data.pos,
            create_graph=self.training,
            retain_graph=True,
        )[0]
        return {"energy": energy, "forces": forces}


def build_model(config: dict[str, Any], dataset: list[Data], device: torch.device) -> PBCEFSHead:
    max_num_elements = int(max(data.atomic_numbers.max().item() for data in dataset)) + 1
    model_cfg = config["model"]
    backbone = PBCeSCNIrrepsBackbone(
        max_num_elements=max_num_elements,
        sphere_channels=model_cfg["sphere_channels"],
        lmax=model_cfg["lmax"],
        mmax=model_cfg["mmax"],
        cutoff=config["cutoff"],
        max_neighbors=model_cfg["max_neighbors"],
        num_layers=model_cfg["num_layers"],
        hidden_channels=model_cfg["hidden_channels"],
        edge_channels=model_cfg["edge_channels"],
        num_distance_basis=model_cfg["num_distance_basis"],
        num_k_x=model_cfg["num_k_x"],
        num_k_y=model_cfg["num_k_y"],
        num_k_z=model_cfg["num_k_z"],
        downprojection=model_cfg["downprojection"],
        num_hidden=model_cfg["num_hidden"],
    )
    return PBCEFSHead(backbone).to(device)


def move_batch(batch: Batch, device: torch.device) -> Batch:
    batch = batch.to(device)
    batch.pos.requires_grad_(True)
    return batch


def compute_metrics(pred_energy, pred_forces, true_energy, true_forces, natoms) -> dict[str, torch.Tensor]:
    natoms = natoms.view(-1, 1).to(pred_energy.dtype)
    energy_error = pred_energy / natoms - true_energy / natoms
    force_error = pred_forces - true_forces
    return {
        "energy_mse_total": torch.mean((pred_energy - true_energy) ** 2),
        "energy_rmse": torch.sqrt(torch.mean(energy_error**2)),
        "energy_mae": torch.mean(torch.abs(energy_error)),
        "force_mse": torch.mean(force_error**2),
        "force_rmse": torch.sqrt(torch.mean(force_error**2)),
        "force_mae": torch.mean(torch.abs(force_error)),
    }


def run_epoch(
    model: PBCEFSHead,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    energy_weight: float,
    force_weight: float,
    grad_clip_norm: float,
    desc: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {key: 0.0 for key in ("loss", "energy_rmse", "energy_mae", "force_rmse", "force_mae")}
    n_batches = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        data = move_batch(batch, device)
        outputs = model(data)
        metrics = compute_metrics(outputs["energy"], outputs["forces"], data.y, data.force, data.natoms)
        loss = energy_weight * metrics["energy_mse_total"] + force_weight * metrics["force_mse"]
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        for key in ("energy_rmse", "energy_mae", "force_rmse", "force_mae"):
            totals[key] += float(metrics[key].detach().cpu())
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError(f"No batches in {desc}")
    return {key: value / n_batches for key, value in totals.items()}


def write_metrics_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def torch_load(path: str | Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def smoke_check(model: PBCEFSHead, train_loader: DataLoader, valid_loader: DataLoader, device: torch.device) -> None:
    model.train()
    for name, loader in (("train", train_loader), ("valid", valid_loader)):
        batch = move_batch(next(iter(loader)), device)
        if batch.cell.ndim != 3 or batch.cell.shape[-2:] != (3, 3):
            raise RuntimeError(f"{name} batch has invalid cell shape {tuple(batch.cell.shape)}")
        if not bool(batch.pbc.all()):
            raise RuntimeError(f"{name} batch does not have all-True PBC")
        k_cell, _ = x_to_k_cell(batch.cell)
        k_index = model.backbone.k_index_product_set.to(device=device, dtype=batch.pos.dtype)
        k_grid = torch.matmul(k_index, k_cell)
        if k_grid.ndim != 3 or k_grid.shape[0] != int(batch.natoms.numel()):
            raise RuntimeError(f"{name} k-grid has invalid shape {tuple(k_grid.shape)}")
        outputs = model(batch)
        loss = outputs["energy"].sum() + outputs["forces"].pow(2).mean()
        loss.backward()
        model.zero_grad(set_to_none=True)
        print(
            f"smoke {name}: cell={tuple(batch.cell.shape)} pbc_all={bool(batch.pbc.all())} "
            f"k_grid={tuple(k_grid.shape)} energy={tuple(outputs['energy'].shape)} forces={tuple(outputs['forces'].shape)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nacl/escn_md_irreps_pbc.yaml",
        help="Path to the YAML training configuration.",
    )
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None, help="Prefix for unique output subdirectories.")
    parser.add_argument("--unique-output-dir", dest="unique_output_dir", action="store_true", default=None)
    parser.add_argument("--no-unique-output-dir", dest="unique_output_dir", action="store_false")
    parser.add_argument("--valid-fraction", type=float, default=None)
    parser.add_argument("--train-fraction", type=float, default=None, help="Use only this fraction of the post-split training set.")
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--force-weight", type=float, default=None)
    parser.add_argument("--energy-weight", type=float, default=None)
    parser.add_argument("--num-k-x", type=int, default=None)
    parser.add_argument("--num-k-y", type=int, default=None)
    parser.add_argument("--num-k-z", type=int, default=None)
    parser.add_argument("--downprojection", type=int, default=None)
    parser.add_argument("--num-hidden", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", default=None, help="Run one train and one valid forward/backward pass only.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for debugging/smoke runs.")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a checkpoint produced by this script.")
    parser.add_argument("--patience", type=int, default=None, help="Stop after this many epochs without validation-loss improvement.")
    parser.add_argument("--min-delta", type=float, default=None, help="Minimum validation-loss decrease counted as improvement.")
    parser.add_argument(
        "--save-best-history",
        action="store_true",
        default=None,
        help="Also keep loss-suffixed best checkpoints. By default only model-best.pth and final are written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = default_config()
    if args.config is not None:
        merge_config(config, load_config(args.config))

    cli_config = {
        "seed": args.seed,
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "unique_output_dir": args.unique_output_dir,
        "run_name": args.run_name,
        "valid_fraction": args.valid_fraction,
        "train_fraction": args.train_fraction,
        "cutoff": args.cutoff,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "force_weight": args.force_weight,
        "energy_weight": args.energy_weight,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "max_frames": args.max_frames,
        "smoke": args.smoke,
        "resume": args.resume,
        "save_best_history": args.save_best_history,
    }
    merge_config(config, {key: value for key, value in cli_config.items() if value is not None})

    cli_model_config = {
        "num_k_x": args.num_k_x,
        "num_k_y": args.num_k_y,
        "num_k_z": args.num_k_z,
        "downprojection": args.downprojection,
        "num_hidden": args.num_hidden,
    }
    merge_config(config["model"], {key: value for key, value in cli_model_config.items() if value is not None}, path="config.model")

    if config["smoke"]:
        config["num_epochs"] = 1

    set_seed(int(config["seed"]))
    output_dir = resolve_output_dir(config)
    config["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_effective_config(output_dir, config)
    logger = setup_logging(output_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("Configuration: %s", config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    dataset = load_dataset(config["data_path"])
    if config["max_frames"] is not None:
        dataset = dataset[: int(config["max_frames"])]
    train_set, valid_set = split_dataset(dataset, config["valid_fraction"], int(config["seed"]))
    if config["train_fraction"] is not None:
        train_fraction = float(config["train_fraction"])
        if not 0.0 < train_fraction <= 1.0:
            raise ValueError("train_fraction must be in (0, 1]")
        train_indices = list(range(len(train_set)))
        random.Random(int(config["seed"])).shuffle(train_indices)
        n_train = max(1, int(round(len(train_indices) * train_fraction)))
        train_set = [train_set[i] for i in train_indices[:n_train]]
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=config["batch_size"], shuffle=False)
    logger.info("Loaded %d frames: train=%d valid=%d", len(dataset), len(train_set), len(valid_set))
    logger.info("First cell: %s", train_set[0].cell.view(3, 3).tolist())
    logger.info("First pbc: %s", train_set[0].pbc.view(3).tolist())

    model = build_model(config, dataset, device)
    logger.info("Trainable parameters: %s", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if config["smoke"]:
        smoke_check(model, train_loader, valid_loader, device)
        logger.info("Smoke check passed")
        return 0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        amsgrad=config["amsgrad"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["lr_step_size"],
        gamma=config["lr_gamma"],
    )
    metrics_csv = output_dir / "metrics.csv"
    best_val_loss = math.inf
    best_path = output_dir / "model-best.pth"
    val_metrics = None
    start_epoch = 0
    epochs_without_improvement = 0

    if config["resume"]:
        checkpoint = torch_load(config["resume"], device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("val_loss", math.inf)))
        logger.info("Resumed from %s at epoch %d with best_val_loss=%.6f", config["resume"], start_epoch, best_val_loss)

    for epoch in range(start_epoch, int(config["num_epochs"])):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            energy_weight=config["energy_weight"],
            force_weight=config["force_weight"],
            grad_clip_norm=config["grad_clip_norm"],
            desc=f"Epoch {epoch + 1} [train]",
        )
        val_metrics = run_epoch(
            model,
            valid_loader,
            device,
            optimizer=None,
            energy_weight=config["energy_weight"],
            force_weight=config["force_weight"],
            grad_clip_norm=config["grad_clip_norm"],
            desc=f"Epoch {epoch + 1} [valid]",
        )
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        write_metrics_row(metrics_csv, row)
        logger.info(
            "Epoch %d/%d train_loss=%.6f val_loss=%.6f val_energy_mae=%.6f val_force_mae=%.6f",
            epoch + 1,
            int(config["num_epochs"]),
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["energy_mae"],
            val_metrics["force_mae"],
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "best_val_loss": min(best_val_loss, val_metrics["loss"]),
            "config": config,
        }
        improved = val_metrics["loss"] < (best_val_loss - float(config["min_delta"]))
        if improved:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(checkpoint, best_path)
            if config["save_best_history"]:
                torch.save(checkpoint, output_dir / f"model-best_valloss{best_val_loss:.6f}.pth")
        else:
            epochs_without_improvement += 1
            logger.info(
                "No validation improvement for %d/%d epochs (best_val_loss=%.6f)",
                epochs_without_improvement,
                int(config["patience"]),
                best_val_loss,
            )
            if epochs_without_improvement >= int(config["patience"]):
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    if val_metrics is None:
        raise RuntimeError("No epochs were run")
    final_epoch = epoch + 1
    final_path = output_dir / f"model-final_epoch{final_epoch}_valloss{val_metrics['loss']:.6f}.pth"
    torch.save(
        {
            "epoch": final_epoch - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "best_val_loss": best_val_loss,
            "config": config,
        },
        final_path,
    )
    logger.info("Final checkpoint saved: %s", final_path)
    logger.info("Final val_energy_mae=%.6f val_force_mae=%.6f", val_metrics["energy_mae"], val_metrics["force_mae"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
