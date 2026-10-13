from __future__ import annotations

import torch
from torch import nn

from .task import input_dims


class MemoryControl(nn.Module):
    """Control never receives stimulus or memory; both transitions are learned."""
    def __init__(self, cfg, memory_dim=None):
        super().__init__()
        n = cfg['n']
        self.memory_dim = memory_dim or cfg['model']['memory_dim']
        self.control_dim = cfg['model']['control_dim']
        stim_dim, ctrl_dim = input_dims(n)
        self.control = nn.GRU(ctrl_dim, self.control_dim, batch_first=True)
        self.memory = nn.GRU(stim_dim + self.control_dim + 5, self.memory_dim, batch_first=True)
        self.readout = nn.Sequential(nn.Linear(self.memory_dim + self.control_dim + n + 2,
                                               cfg['model']['readout_dim']), nn.ReLU(),
                                     nn.Linear(cfg['model']['readout_dim'], n))

    def states(self, x, m=None, c=None):
        ctrl, _ = self.control(x['ctrl'], None if c is None else c.unsqueeze(0).contiguous())
        mem, _ = self.memory(torch.cat((x['stim'], ctrl, x['phase']), dim=-1),
                             None if m is None else m.unsqueeze(0).contiguous())
        row = torch.arange(len(ctrl), device=ctrl.device)
        index = x['lengths'] - 1
        return mem[row, index], ctrl[row, index]

    def forward(self, x, query, m=None, c=None):
        m, c = self.states(x, m, c)
        return self.readout(torch.cat((m, c, query), dim=-1))
