import numpy as np
import torch
from torch.utils.data import Dataset

class InSARDataset(Dataset):
    def __init__(self, vel1_patches, vel2_patches, normalize=True):
        self.vel1 = vel1_patches
        self.vel2 = vel2_patches
        self.delta = [v2 - v1 for v1, v2 in zip(self.vel1, self.vel2)]
        self.normalize = normalize
        if normalize and len(self.vel2) > 0:
            concat = np.concatenate([p.flatten() for p in self.vel2])
            self.mean = float(np.mean(concat))
            self.std = float(np.std(concat) + 1e-6)
        else:
            self.mean, self.std = 0.0, 1.0

    def __len__(self):
        return len(self.vel1)

    def __getitem__(self, idx):
        v2 = (self.vel2[idx] - self.mean) / self.std if self.normalize else self.vel2[idx]
        d  = self.delta[idx]
        v2 = torch.tensor(v2, dtype=torch.float32).unsqueeze(0)
        d  = torch.tensor(d, dtype=torch.float32).unsqueeze(0)
        return v2, d
