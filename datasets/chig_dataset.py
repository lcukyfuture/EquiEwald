# -*- coding: utf-8 -*-
"""
AIMD-Chig dataset parser with kcal/mol units (no energy shift)

Simple version for training with chig_train_escn_multigpu.py
- Energy: Hartree -> kcal/mol (1 Hartree = 627.509474 kcal/mol)
- Forces: Hartree/Å -> kcal/(mol·Å)
- No energy shift applied
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import re
import numpy as np

import torch
from torch_geometric.data import Data as PyGData


# --------- Element Mapping ---------
_ELEMENT_TO_Z = {
    "H": 1, "C": 6, "N": 7, "O": 8
}

def elements_to_atomic_numbers(elements: List[str]) -> np.ndarray:
    """Convert element symbols to atomic numbers"""
    z = []
    for e in elements:
        if e not in _ELEMENT_TO_Z:
            raise KeyError(f"Unknown element symbol '{e}'. Supported: {list(_ELEMENT_TO_Z.keys())}")
        z.append(_ELEMENT_TO_Z[e])
    return np.asarray(z, dtype=np.int64)


# --------- Energy Extraction ---------
def extract_energy_from_comment(comment_line: str) -> Optional[float]:
    """
    Extract energy (Hartree) from comment line.
    Example: "# ORCA AIMD Position Step 26, t=26.00 fs, E_Pot=-4511.60955077 Hartree"
    """
    # Try E_Pot pattern
    m = re.search(r"[Ee]\s*[_ ]?\s*Pot\s*=\s*([\-\d\.Ee\+]+)", comment_line)
    if m:
        try:
            return float(m.group(1))
        except:
            pass
    
    # Try Energy pattern
    m = re.search(r"[Ee]nergy\s*[:=]\s*([\-\d\.Ee\+]+)", comment_line)
    if m:
        try:
            return float(m.group(1))
        except:
            pass
    
    # Fallback: first float
    m = re.search(r"([\-+]?\d+\.\d+(?:[eE][\-+]?\d+)?)", comment_line)
    if m:
        try:
            return float(m.group(1))
        except:
            pass
    
    return None


# --------- XYZ Frame Indexer ---------
class MultiFrameXYZIndex:
    """Build byte offsets for multi-frame XYZ file"""
    
    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(str(self.path))
        self._offsets: List[int] = []
        self._build_index()
    
    def _build_index(self):
        """Scan file and record byte offsets of each frame"""
        offsets = []
        with self.path.open("rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                line_str = line.decode(errors="ignore").strip()
                if not line_str:
                    continue
                
                # First line should be atom count
                try:
                    n = int(line_str.split()[0])
                except:
                    continue
                
                # Read comment line
                comment = f.readline()
                if not comment:
                    break
                
                # Record frame start position
                offsets.append(pos)
                
                # Skip N atom lines
                for _ in range(n):
                    if not f.readline():
                        break
        
        self._offsets = offsets
    
    def __len__(self):
        return len(self._offsets)
    
    def get_frame(self, i: int) -> Tuple[List[str], np.ndarray, Optional[float], str]:
        """
        Get frame i: (elements, values, energy, comment)
        - elements: list of element symbols
        - values: (N, 3) array of positions or forces
        - energy: float (Hartree) or None
        - comment: comment line string
        """
        if i < 0 or i >= len(self._offsets):
            raise IndexError(f"Frame {i} out of range [0, {len(self._offsets)})")
        
        with self.path.open("rb") as f:
            f.seek(self._offsets[i])
            
            # Line 1: atom count
            n_line = f.readline().decode(errors="ignore").strip()
            n = int(n_line.split()[0])
            
            # Line 2: comment
            comment = f.readline().decode(errors="ignore").rstrip("\n")
            energy = extract_energy_from_comment(comment)
            
            # Next N lines: atom data
            elems: List[str] = []
            vals = np.zeros((n, 3), dtype=np.float64)
            
            for j in range(n):
                parts = f.readline().decode(errors="ignore").strip().split()
                if len(parts) == 0:
                    raise ValueError(f"Empty atom line at frame {i}, atom {j}")
                
                elem = parts[0]
                floats = [float(x) for x in parts[1:] if re.match(r'^[-+]?\d+(\.\d+)?([eE][-+]?\d+)?$', x)]
                
                if len(floats) < 3:
                    raise ValueError(f"Need 3 floats, got {parts} at frame {i}, atom {j}")
                
                vals[j, :] = floats[-3:]
                elems.append(elem)
        
        return elems, vals, energy, comment


# --------- Dataset ---------
class AIMDChigDataset:
    """
    AIMD-Chig dataset with kcal/mol units and no energy shift.
    
    Returns PyG Data objects with:
        - y: [1] energy in kcal/mol
        - pos: [N, 3] positions in Å
        - force: [N, 3] forces in kcal/(mol·Å)
        - atomic_numbers: [N] atomic numbers
        - natoms: [1] number of atoms
    
    Args:
        root: Dataset root directory
        split_npz: Split file name (e.g., "scaffold.npz")
        split_key: Split key (e.g., "train_id", "vali_id", "test_id")
        validate_data: Whether to filter invalid items
        use_cache: Whether to cache processed data
    """
    
    # Conversion factor: Hartree -> kcal/mol
    HARTREE_TO_KCAL = 627.509474
    
    def __init__(
        self,
        root: str,
        split_npz: str = "scaffold.npz",
        split_key: str = "train_id",
        validate_data: bool = True,
        use_cache: bool = True
    ):
        self.root = Path(root)
        self.split_npz = self.root / "Split" / split_npz
        self.split_key = split_key
        self.use_cache = use_cache
        
        if not self.split_npz.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_npz}")
        
        # Setup cache
        self.processed_dir = self.root / "processed_kcal"
        self.processed_dir.mkdir(exist_ok=True)
        
        split_name = Path(split_npz).stem
        self.cache_file = self.processed_dir / f"{split_name}_{split_key}_kcal.pt"
        
        # Load or process
        if self.use_cache and self.cache_file.exists():
            print(f"Loading cached data from {self.cache_file}")
            self._load_from_cache()
        else:
            print(f"Processing {split_key} from raw data...")
            self._process_from_raw(validate_data)
            if self.use_cache:
                print(f"Saving cache to {self.cache_file}")
                self._save_to_cache()
        
        print(f"Dataset ready: {len(self.data_list)} samples")
    
    def _process_from_raw(self, validate_data: bool):
        """Process raw XYZ files"""
        split = np.load(self.split_npz)
        if self.split_key not in split.files:
            raise KeyError(f"Split key '{self.split_key}' not found in {self.split_npz.name}")
        
        raw_items = split[self.split_key]
        print(f"  Raw items: {len(raw_items)}")
        
        # Filter valid items
        if validate_data:
            items = self._filter_valid_items(raw_items)
            print(f"  Valid items: {len(items)} (filtered {len(raw_items) - len(items)})")
        else:
            items = raw_items
        
        # Process all items
        self.data_list = []
        self._coord_idx_cache: Dict[Tuple[int, int], MultiFrameXYZIndex] = {}
        self._force_idx_cache: Dict[Tuple[int, int], MultiFrameXYZIndex] = {}
        
        print(f"  Processing {len(items)} samples...")
        for i, (sf, anchor, frame) in enumerate(items):
            if (i + 1) % 1000 == 0:
                print(f"    {i + 1}/{len(items)}")
            
            try:
                data = self._process_single_item(int(sf), int(anchor), int(frame))
                self.data_list.append(data)
            except Exception as e:
                print(f"    Warning: Failed item {i} (sf={sf}, anchor={anchor}, frame={frame}): {e}")
                continue
        
        # Clear caches
        self._coord_idx_cache.clear()
        self._force_idx_cache.clear()
        
        print(f"  Successfully processed: {len(self.data_list)} samples")
    
    def _process_single_item(self, sf: int, anchor: int, frame: int) -> PyGData:
        """Process a single frame and return PyG Data"""
        # Get indexers
        coord_idx, force_idx = self._get_indexers(sf, anchor)
        
        # Read frame data
        elems_c, pos, energy, _ = coord_idx.get_frame(frame)
        elems_f, force, _, _ = force_idx.get_frame(frame)
        
        if elems_c != elems_f:
            raise ValueError("Element mismatch between coord and force files")
        
        # Convert elements to atomic numbers
        z = elements_to_atomic_numbers(elems_c)
        
        # Unit conversion: Hartree -> kcal/mol
        if energy is None:
            raise ValueError(f"Missing energy for sf={sf}, anchor={anchor}, frame={frame}")
        
        energy_kcal = energy * self.HARTREE_TO_KCAL
        force_kcal = force * self.HARTREE_TO_KCAL
        
        # Create PyG Data
        data = PyGData(
            y=torch.tensor([energy_kcal], dtype=torch.float32),
            pos=torch.tensor(pos, dtype=torch.float32),
            force=torch.tensor(force_kcal, dtype=torch.float32),
            atomic_numbers=torch.tensor(z, dtype=torch.long),
            natoms=torch.tensor([len(z)], dtype=torch.long)
        )
        
        return data
    
    def _get_indexers(self, sf: int, anchor: int) -> Tuple[MultiFrameXYZIndex, MultiFrameXYZIndex]:
        """Get or create indexers for coord and force files"""
        coord_path = self.root / "Coordinates" / str(sf) / f"Coordinates_{anchor}.xyz"
        force_path = self.root / "Force" / str(sf) / f"Force_{anchor}.xyz"
        
        key = (sf, anchor)
        
        if key not in self._coord_idx_cache:
            self._coord_idx_cache[key] = MultiFrameXYZIndex(coord_path)
        
        if key not in self._force_idx_cache:
            self._force_idx_cache[key] = MultiFrameXYZIndex(force_path)
        
        return self._coord_idx_cache[key], self._force_idx_cache[key]
    
    def _filter_valid_items(self, raw_items: np.ndarray) -> np.ndarray:
        """Filter out invalid items"""
        valid_items = []
        
        for sf, anchor, frame in raw_items:
            sf, anchor, frame = int(sf), int(anchor), int(frame)
            
            try:
                coord_path = self.root / "Coordinates" / str(sf) / f"Coordinates_{anchor}.xyz"
                force_path = self.root / "Force" / str(sf) / f"Force_{anchor}.xyz"
                
                if not coord_path.exists() or not force_path.exists():
                    continue
                
                coord_idx = MultiFrameXYZIndex(coord_path)
                force_idx = MultiFrameXYZIndex(force_path)
                
                if frame >= len(coord_idx) or frame >= len(force_idx):
                    continue
                
                valid_items.append([sf, anchor, frame])
                
            except:
                continue
        
        return np.array(valid_items) if valid_items else np.array([])
    
    def _save_to_cache(self):
        """Save processed data to cache"""
        torch.save({
            'data_list': self.data_list,
            'split_key': self.split_key,
        }, self.cache_file)
    
    def _load_from_cache(self):
        """Load processed data from cache"""
        cache = torch.load(self.cache_file, weights_only=False)
        self.data_list = cache['data_list']
        
        if cache.get('split_key') != self.split_key:
            print(f"Warning: Cached split_key mismatch")
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, i: int) -> PyGData:
        if i < 0 or i >= len(self.data_list):
            raise IndexError(f"Index {i} out of range")
        return self.data_list[i]


if __name__ == "__main__":
    # Test
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--split", type=str, default="scaffold.npz")
    parser.add_argument("--key", type=str, default="train_id")
    args = parser.parse_args()
    
    ds = AIMDChigDataset(args.root, args.split, args.key)
    
    print(f"\nDataset size: {len(ds)}")
    
    if len(ds) > 0:
        sample = ds[0]
        print(f"\nFirst sample:")
        print(f"  Energy: {sample.y.item():.2f} kcal/mol")
        print(f"  Atoms: {sample.natoms.item()}")
        print(f"  Pos shape: {sample.pos.shape}")
        print(f"  Force shape: {sample.force.shape}")
        print(f"  Elements: {sample.atomic_numbers.tolist()}")
