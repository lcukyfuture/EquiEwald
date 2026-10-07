"""Small CPU checks; these do not reproduce benchmark accuracy."""
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def make_molecule(n=4):
    from torch_geometric.data import Data
    return Data(
        pos=torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.2, 0.1],
                          [0.2, 1.3, 0.4], [-0.3, 0.1, 1.5], [1.8, 1.4, 0.2]])[:n],
        atomic_numbers=torch.tensor([6, 1, 8, 1, 1])[:n],
        natoms=torch.tensor([n]), y=torch.tensor([0.2]),
        force=torch.zeros(n, 3),
    )


def check_step(model, batch, forward):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    energy, force = forward(batch)
    assert energy.shape == batch.y.shape
    assert force.shape == batch.pos.shape
    assert torch.isfinite(energy).all() and torch.isfinite(force).all()
    loss = (energy - batch.y).square().mean() + (force - batch.force).square().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    optimizer.step()


def test_bundled_dimer_reader(tmp_path):
    from datasets.dimer_dataset import DimerDataset
    cfg = yaml.safe_load((ROOT / 'configs/dimer/escn_md_irreps.yaml').read_text())
    raw = tmp_path / 'raw'
    raw.mkdir()
    for key, split in [('train_xyz', 'train'), ('val_xyz', 'val')]:
        filename = cfg['data'][key]
        shutil.copyfile(ROOT / 'data/dimer/raw' / filename, raw / filename)
        ds = DimerDataset(str(tmp_path), filename, split=split)
        assert len(ds) >= 2
        batch = Batch.from_data_list([ds[0], ds[1]])
        assert batch.y.shape == (2,)
        assert batch.force.shape == batch.pos.shape
        assert torch.isfinite(batch.y).all() and torch.isfinite(batch.force).all()


@pytest.mark.parametrize('ewald', [False, True])
def test_nonperiodic_energy_force_backward_and_reload(ewald, tmp_path):
    from fairchem.core.models.uma.escn_md_irreps import eSCNMDBackbone, MLP_EFS_Head
    from train_dimer import convert_to_escn_format
    torch.manual_seed(7)
    backbone = eSCNMDBackbone(
        max_num_elements=10, sphere_channels=8, hidden_channels=8, edge_channels=8,
        lmax=3, mmax=2, num_layers=1, num_distance_basis=16, cutoff=3.0,
        max_neighbors=20, dataset_list=['dimer'], irreps=ewald,
        ewald_hyperparams=(dict(k_cutoff=0.6, delta_k=0.3, num_k_rbf=8,
                               downprojection=4, num_hidden=0) if ewald else None),
    )
    head = MLP_EFS_Head(backbone)
    model = torch.nn.ModuleList([backbone, head])
    def forward(batch):
        batch = convert_to_escn_format(batch)
        out = head(batch, backbone(batch))
        return out['energy']['energy'], out['forces']['forces']
    batch = Batch.from_data_list([make_molecule(4), make_molecule(5)])
    check_step(model, batch, forward)
    path = tmp_path / 'model.pt'
    torch.save(model.state_dict(), path)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    energy, forces = forward(batch.clone())
    assert torch.isfinite(energy).all() and torch.isfinite(forces).all()


def test_periodic_energy_force_backward():
    from train_nacl import atoms_to_data, build_model, default_config, move_batch
    torch.manual_seed(11)
    atoms = Atoms('NaClNaCl', positions=[[0.2, 0.2, 0.2], [2.0, 0.2, 0.2],
                                        [0.4, 2.2, 0.3], [2.3, 2.1, 0.4]],
                  cell=[5.5, 5.5, 5.5], pbc=True)
    atoms.calc = SinglePointCalculator(atoms, energy=-1.0, forces=np.zeros((4, 3)))
    data = atoms_to_data(atoms)
    config = default_config()
    config['cutoff'] = 3.0
    config['model'].update(sphere_channels=8, hidden_channels=8, edge_channels=8,
                           lmax=3, mmax=2, num_layers=1, num_distance_basis=16,
                           num_k_x=1, num_k_y=1, num_k_z=1, downprojection=4, num_hidden=0)
    model = build_model(config, [data], torch.device('cpu'))
    batch = move_batch(Batch.from_data_list([data, data.clone()]), torch.device('cpu'))
    def forward(batch):
        out = model(batch)
        return out['energy'], out['forces']
    check_step(model, batch, forward)


@pytest.mark.parametrize('num_k_z', [1, 3])
def test_oc20_periodic_energy_force_backward(num_k_z):
    from ocpmodels.models.escn.escn import eSCN
    torch.manual_seed(17)
    model = eSCN(
        num_atoms=None, bond_feat_dim=None, num_targets=1,
        use_pbc=True, regress_forces=True, otf_graph=True,
        max_neighbors=20, cutoff=3.0, max_num_elements=20,
        num_layers=1, lmax_list=[2], mmax_list=[2], sphere_channels=8,
        hidden_channels=8, edge_channels=8, num_sphere_samples=16,
        distance_resolution=0.5, use_ewald=True,
        ewald_hyperparams=dict(num_k_x=1, num_k_y=1, num_k_z=num_k_z,
                              downprojection_size=4, num_hidden=0),
    )
    data = make_molecule(4)
    data.cell = torch.eye(3).unsqueeze(0) * 5.5
    data.pbc = torch.ones(1, 3, dtype=torch.bool)
    batch = Batch.from_data_list([data, data.clone()])
    check_step(model, batch, model)


def test_chignolin_lightning_forward_backward():
    from train_chig import ChigESCNLightningModule
    config = yaml.safe_load((ROOT / 'configs/chig/chig_escn.yaml').read_text())
    config['model'].update(sphere_channels=8, hidden_channels=8, edge_channels=8,
                           num_layers=1, irreps=True)
    config['ewald_hyperparams'].update(k_cutoff=0.6, delta_k=0.3, num_k_rbf=8,
                                       downprojection=4, num_hidden=0)
    model = ChigESCNLightningModule(config, task_mean=0.0, task_std=1.0)
    batch = Batch.from_data_list([make_molecule(4), make_molecule(5)])
    check_step(model, batch, lambda batch: model(batch)[:2])
