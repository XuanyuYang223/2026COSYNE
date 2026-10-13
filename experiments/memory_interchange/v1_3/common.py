from pathlib import Path
import json
import time
import torch
import yaml
from ..common import ROOT as BASE, write_json

ROOT = Path(__file__).resolve().parent
RESULTS = BASE / 'results/v1_3'
RUNS = BASE / 'runs/v1_3'

def config():
    return yaml.safe_load((ROOT / 'revision_v1_3.yaml').read_text())

class Budget:
    def __init__(self):
        self.path = RESULTS / 'budget.json'
        self.inherited = json.loads((RESULTS / 'preserved_versions.json').read_text())['inherited_gpu_seconds']
        self.limit = config()['execution']['gpu_hour_budget'] * 3600
        self.start = None
    @property
    def used(self):
        return json.loads(self.path.read_text())['gpu_seconds'] if self.path.exists() else self.inherited
    def remaining(self):
        return self.limit - self.used - (time.perf_counter()-self.start if self.start is not None else 0)
    def check(self, reserve=0):
        if self.remaining() <= reserve:
            raise RuntimeError('12 GPU-hour total budget exhausted; no automatic extension')
    def __enter__(self):
        self.check()
        if torch.cuda.is_available(): self.start = time.perf_counter()
        return self
    def __exit__(self, *args):
        if self.start is not None:
            torch.cuda.synchronize()
            total = self.used + time.perf_counter()-self.start
            write_json(self.path, {'gpu_seconds': total, 'inherited_gpu_seconds': self.inherited,
                                  'v1_3_gpu_seconds': total-self.inherited, 'limit_seconds': self.limit})
        self.start = None
