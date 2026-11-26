import torch
from torch.utils.data import Dataset
from safetensors import safe_open
from .utils import LABELS

class EmbDataset(Dataset):
    def __init__(self, safetensors_path, metadata, indices):
        """
        safetensors_path: Path to the .safetensors file
        metadata: The dictionary loaded from _metadata.json
        indices: List of string protein_ids belonging to this fold split
        """
        self.path = safetensors_path
        self.metadata = metadata
        self.idx = list(indices)
        
    def __len__(self): 
        return len(self.idx)
        
    def __getitem__(self, i):
        pid = self.idx[i]
        meta = self.metadata[pid]
        
        with safe_open(self.path, framework="pt", device="cpu") as f:
            embedding = f.get_tensor(f"emb_{pid}").float()
            
        return embedding, int(meta["label"]), pid

def collate(batch):
    xs, ys, ids = zip(*batch)
    L = max(x.size(0) for x in xs)
    D = xs[0].size(1)
    x = torch.zeros(len(xs), L, D)
    m = torch.zeros(len(xs), L, dtype=torch.bool)
    for i, a in enumerate(xs): 
        x[i, :len(a)] = a
        m[i, :len(a)] = 1
    return x, m, torch.tensor(ys), list(ids)

def label_id(x):
    if str(x).strip() not in LABELS: 
        raise ValueError(f"Unknown label {x}; use {list(LABELS)}")
    return LABELS[str(x).strip()]