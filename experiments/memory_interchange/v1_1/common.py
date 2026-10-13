from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import yaml

from ..common import ROOT as BASE, write_json

ROOT = Path(__file__).resolve().parent
RESULTS = BASE / 'results' / 'v1_1'
RUNS = BASE / 'runs' / 'v1_1'


def config():
    return yaml.safe_load((ROOT / 'config.yaml').read_text())


class Budget:
    """One immutable inherited cost plus v1.1 active-call conservative wall time."""
    def __init__(self):
        self.path = RESULTS / 'budget.json'
        self.inherited = json.loads((RESULTS / 'preserved_v1_0.json').read_text())['inherited_gpu_seconds']
        self.limit = config()['execution']['gpu_hour_budget'] * 3600
        self.start = None

    @property
    def used(self):
        return json.loads(self.path.read_text())['gpu_seconds'] if self.path.exists() else self.inherited

    def remaining(self):
        return self.limit - self.used - (time.perf_counter()-self.start if self.start is not None else 0)

    def check(self, reserve=0):
        if self.remaining() <= reserve:
            raise RuntimeError('12 GPU-hour total budget including v1.0 exhausted')

    def __enter__(self):
        self.check()
        if torch.cuda.is_available():
            self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.start is not None:
            total = self.used + time.perf_counter()-self.start
            write_json(self.path, {'gpu_seconds': total, 'v1_0_gpu_seconds': self.inherited,
                                  'v1_1_gpu_seconds': total-self.inherited, 'limit_seconds': self.limit})
        self.start = None


def model_config(revision, stage=None):
    timing = revision['timings'][stage['timing'] if stage else 'original']
    return {'n': revision['n'], 'model': revision['model'], 'timing': timing,
            'execution': {'cpu_threads': revision['execution']['cpu_threads'],
                          'deterministic': revision['execution']['deterministic'], 'cache_batch_size': 1024}}
