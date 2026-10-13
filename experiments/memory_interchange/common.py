from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent


def config(path=None):
    with open(path or ROOT / 'config.yaml') as f:
        return yaml.safe_load(f)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed, threads=4, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)


def rng_state(rng):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'generator': rng.bit_generator.state, 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, rng):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    rng.bit_generator.state = state['generator']
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Budget:
    """Persistent total active GPU wall-time; CPU work is excluded."""
    def __init__(self, path, hours):
        self.path = Path(path)
        self.limit = hours * 3600
        self.started = None

    @property
    def used(self):
        return json.loads(self.path.read_text()).get('gpu_seconds', 0.0) if self.path.exists() else 0.0

    def remaining(self):
        current = time.perf_counter() - self.started if self.started is not None else 0
        return self.limit - self.used - current

    def check(self, reserve=0):
        if self.remaining() <= reserve:
            raise RuntimeError('Predefined 12 GPU-hour budget exhausted; no automatic extension.')

    def __enter__(self):
        self.check()
        self.started = time.perf_counter() if torch.cuda.is_available() else None
        return self

    def __exit__(self, *args):
        if self.started is not None:
            elapsed = time.perf_counter() - self.started
            write_json(self.path, {'gpu_seconds': self.used + elapsed, 'limit_seconds': self.limit})
        self.started = None
