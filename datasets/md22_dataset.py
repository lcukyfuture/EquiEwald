"""
MD22 Dataset for eSCN-MD Model
This dataset is specifically designed to match eSCN-MD and AIMNet2 input requirements
"""
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from tqdm import tqdm


class MD22_ESCN(InMemoryDataset):
    """
    MD22 Dataset optimized for eSCN-MD model.
    
    Key features:
    - Supports train/val/test splits
    - Provides all required fields for eSCN-MD and AIMNet2
    - Handles charge, spin, and dataset embeddings
    - Compatible with both PBC and non-PBC systems
    
    Args:
        root: Root directory for dataset
        dataset_arg: Molecule name (e.g., "DHA", "AT_AT")
        split: 'train', 'val', or 'test'
        split_ratio: Train/val/test ratio (default: 0.95/0.05/0.0 for train+val split)
        seed: Random seed for reproducibility
        transform: Optional transform
        pre_transform: Optional pre-transform
    """
    
    def __init__(
        self,
        root,
        dataset_arg: str,
        split: str = 'train',
        split_ratio=(0.95, 0.05, 0.0),
        seed: int = 0,
        transform=None,
        pre_transform=None,
    ):
        assert split in {'train', 'val', 'test'}, f"Split must be train/val/test, got {split}"
        
        self.dataset_arg = dataset_arg
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed

        super().__init__(osp.join(root, dataset_arg), transform, pre_transform)
        # PyTorch 2.6+ requires weights_only=False for custom classes
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def molecule_names(self):
        """Mapping from molecule keys to NPZ filenames"""
        return dict(
            Ac_Ala3_NHMe="md22_Ac-Ala3-NHMe.npz",
            DHA="md22_DHA.npz",
            stachyose="md22_stachyose.npz",
            AT_AT="md22_AT-AT.npz",
            AT_AT_CG_CG="md22_AT-AT-CG-CG.npz",
            buckyball_catcher="md22_buckyball-catcher.npz",
            double_walled_nanotube="md22_dw_nanotube.npz",
        )

    @property
    def raw_file_names(self):
        if self.dataset_arg not in self.molecule_names:
            raise ValueError(
                f"Unknown molecule '{self.dataset_arg}'. "
                f"Available: {list(self.molecule_names.keys())}"
            )
        return [self.molecule_names[self.dataset_arg]]

    @property
    def processed_file_names(self):
        """Include split, seed, and trainval_count in filename to avoid conflicts"""
        count = self.molecule_splits[self.dataset_arg]
        return [f"md22_escn_{self.dataset_arg}_{self.split}_seed{self.seed}_tv{count}.pt"]

    @property
    def base_url(self):
        return "http://www.quantum-machine.org/gdml/data/npz/"

    def download(self):
        """Download MD22 data from quantum-machine.org"""
        download_url(
            self.base_url + self.molecule_names[self.dataset_arg], 
            self.raw_dir
        )

    @property
    def molecule_splits(self):
        """
        Recommended train+val sample counts from MD22 paper.
        Remaining samples are used for testing.
        """
        return dict(
            Ac_Ala3_NHMe=6000,
            DHA=8000,
            stachyose=8000,
            AT_AT=3000,
            AT_AT_CG_CG=2000,
            buckyball_catcher=600,
            double_walled_nanotube=800,
        )

    def process(self):
        """
        Process raw MD22 data and create train/val/test splits.
        
        Data format (units as in original MD22 dataset):
        - z: atomic numbers [num_atoms]
        - R: positions [num_structures, num_atoms, 3] in Angstrom
        - E: energies [num_structures] in kcal/mol
        - F: forces [num_structures, num_atoms, 3] in kcal/mol/Angstrom
        """
        # Load raw NPZ data
        path = self.raw_paths[0]
        arr = np.load(path)
        
        z = torch.as_tensor(arr["z"], dtype=torch.long)      # [num_atoms]
        R = torch.as_tensor(arr["R"], dtype=torch.float32)   # [num_structures, num_atoms, 3]
        E = torch.as_tensor(arr["E"], dtype=torch.float32)   # [num_structures]
        F = torch.as_tensor(arr["F"], dtype=torch.float32)   # [num_structures, num_atoms, 3]

        n_total = R.shape[0]
        n_atoms = z.shape[0]
        trainval_count = self.molecule_splits[self.dataset_arg]
        
        if trainval_count > n_total:
            raise ValueError(
                f"train+val count {trainval_count} exceeds total {n_total} "
                f"for {self.dataset_arg}"
            )

        # Create random permutation with fixed seed
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n_total)

        # Split into train+val and test
        idx_trainval = perm[:trainval_count]
        idx_test = perm[trainval_count:]

        # Further split train+val according to split_ratio
        tv_train_ratio = float(self.split_ratio[0])
        tv_val_ratio = float(self.split_ratio[1])
        tv_sum = tv_train_ratio + tv_val_ratio
        
        if tv_sum <= 0:
            raise ValueError("split_ratio must allocate >0 to train+val")
        
        n_train = int(round(trainval_count * tv_train_ratio / tv_sum))
        n_val = trainval_count - n_train

        idx_train = idx_trainval[:n_train]
        idx_val = idx_trainval[n_train:]

        # Select indices for current split
        if self.split == 'train':
            indices = idx_train
        elif self.split == 'val':
            indices = idx_val
        else:  # test
            indices = idx_test

        # Pre-allocate common tensors (only what's needed)
        natoms = torch.tensor([n_atoms], dtype=torch.long)
        
        # AIMNet2 specific fields (per-molecule properties)
        charge = torch.tensor([0.0], dtype=torch.float32)  # Neutral molecule
        spin = torch.tensor([0.0], dtype=torch.float32)    # Singlet state (mult=1)
        
        # Process each structure
        samples = []
        for i in tqdm(indices, desc=f"Processing MD22[{self.dataset_arg}:{self.split}]"):
            pos = R[i]          # [num_atoms, 3]
            energy = E[i:i+1]   # [1] - keep as 1D tensor
            forces = F[i]       # [num_atoms, 3]

            # Create Data object with only required fields
            data = Data(
                # Core structure data
                pos=pos,                    # [num_atoms, 3] - atomic positions (Angstrom)
                atomic_numbers=z,           # [num_atoms] - atomic numbers (Z)
                
                # Targets (MD22 original units: kcal/mol and kcal/mol/Angstrom)
                y=energy,                   # [1] - total energy (kcal/mol)
                force=forces,               # [num_atoms, 3] - forces (kcal/mol/Angstrom)
                
                # Required metadata
                natoms=natoms,              # [1] - number of atoms per molecule
                
                # AIMNet2 specific (per-molecule properties)
                charge=charge.clone(),      # [1] - molecular charge (0 = neutral)
                spin=spin.clone(),          # [1] - spin multiplicity (1 = singlet)
            )

            # Apply filters and transforms
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            samples.append(data)

        # Collate all samples
        data, slices = self.collate(samples)
        
        # Save processed data
        torch.save((data, slices), self.processed_paths[0])
        
        print(f"\n✓ Processed {len(samples)} samples for {self.dataset_arg} ({self.split} split)")
        print(f"  - Train+Val total: {trainval_count}")
        print(f"  - Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")
        print(f"  - Atoms per structure: {n_atoms}")

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'molecule={self.dataset_arg}, '
            f'split={self.split}, '
            f'size={len(self)}, '
            f'seed={self.seed})'
        )


# Convenience function for loading datasets
def load_md22_escn(root, molecule, split='train', seed=0, batch_size=4, num_workers=0):
    """
    Convenience function to load MD22 dataset for eSCN-MD.
    
    Example:
        train_loader = load_md22_escn('./data', 'DHA', split='train', batch_size=4)
        for batch in train_loader:
            # batch is ready for eSCN-MD
            pass
    """
    from torch_geometric.loader import DataLoader
    
    dataset = MD22_ESCN(root, molecule, split=split, seed=seed)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    
    return loader
