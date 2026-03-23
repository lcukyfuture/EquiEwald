"""
Dimer Dataset for eSCN-MD Model
Reads XYZ files (charged dimer interaction data) and creates PyG InMemoryDataset.
"""
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import ase.io


class DimerDataset(InMemoryDataset):
    """
    Charged Dimer Dataset for eSCN-MD model.

    Reads XYZ files using ASE. Target energy is ``inter_energy`` (interaction
    energy between fragments) and target forces are ``forces``.

    Args:
        root: Root directory containing a ``raw/`` subdirectory with XYZ files
        xyz_file: XYZ filename inside ``raw/`` (e.g., "train-id0.xyz")
        split: Split name used for processed file naming ("train", "val", "test")
        transform: Optional transform
        pre_transform: Optional pre-transform
    """

    def __init__(
        self,
        root,
        xyz_file: str,
        split: str = "train",
        transform=None,
        pre_transform=None,
    ):
        assert split in {"train", "val", "test"}, f"Split must be train/val/test, got {split}"

        self.xyz_file = xyz_file
        self.split = split

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [self.xyz_file]

    @property
    def processed_file_names(self):
        name = osp.splitext(self.xyz_file)[0]
        return [f"dimer_{name}_{self.split}.pt"]

    def download(self):
        pass

    def process(self):
        """Process XYZ file into PyG Data objects."""
        path = self.raw_paths[0]
        atoms_list = ase.io.read(path, index=":")

        samples = []
        for atoms in tqdm(atoms_list, desc=f"Processing dimer [{self.split}]"):
            pos = torch.as_tensor(atoms.get_positions(), dtype=torch.float32)
            atomic_numbers = torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long)
            energy = torch.tensor([atoms.info["inter_energy"]], dtype=torch.float32)
            force = torch.as_tensor(atoms.arrays["forces"], dtype=torch.float32)
            natoms = torch.tensor([len(atoms)], dtype=torch.long)

            data = Data(
                pos=pos,
                atomic_numbers=atomic_numbers,
                y=energy,
                force=force,
                natoms=natoms,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            samples.append(data)

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])

        print(f"\nProcessed {len(samples)} samples for dimer ({self.split} split)")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"split={self.split}, "
            f"size={len(self)})"
        )
