import json
import os
import random
from dataclasses import dataclass

import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value, n=1):
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.total / max(self.count, 1)
