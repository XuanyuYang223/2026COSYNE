import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from ...common import seed_all
from ...model import MemoryControl
from ...task import tensor_inputs
from ..audit import gate
from ..common import config, model_config
from ..data import generate, oracle
from ..trace import eager_forward, finite_difference, gradient_path, update_gates
from ..train import initialize, optimizer_for, restore_checkpoint, save_checkpoint, train_step


def test_actual_collate_oracle_and_short_mid_original_padding():
    cfg = config()
    pool = np.array([np.roll(np.arange(7), i) for i in range(7)])
    for stage in cfg['stages']:
        x = generate(pool, 64, np.random.default_rng(13), cfg, stage, True)
        assert np.array_equal(oracle(x), x['target'])
        assert set(x['lengths']) == ({9} if stage['timing'] == 'short' else {23} if stage['timing'] == 'mid' else {37, 47})
    broken = copy.deepcopy(x)
    broken['target'][0] = (broken['target'][0]+1)%7
    assert not np.array_equal(oracle(broken), broken['target'])
    broken = copy.deepcopy(x)
    broken['lengths'][0] -= 1
    broken['phase'][0, broken['lengths'][0]:] = 0
    broken['ctrl'][0, broken['lengths'][0]:] = 0
    broken['stim'][0, broken['lengths'][0]:] = 0
    with pytest.raises(AssertionError):
        oracle(broken)


def test_true_carried_state_gradients_and_short_derivative():
    cfg = config()
    seed_all(5)
    model = MemoryControl(model_config(cfg))
    pool = np.array([np.arange(7), np.arange(7)[::-1]])
    x = generate(pool, 4, np.random.default_rng(12), cfg, cfg['stages'][2])
    result = gradient_path(model, x)
    assert not result['stimulus_gradient_none']
    assert all(v['gradient_none_rows'] == 0 for v in result['events'].values())
    short = generate(pool, 1, np.random.default_rng(12), cfg, cfg['stages'][0])
    assert finite_difference(model, short)['passed']


def test_unique_initialization_changes_only_effective_update_bias():
    cfg = config()
    a = initialize(cfg, 'A').cpu()
    b = initialize(cfg, 'B').cpu()
    for name, original in a.state_dict().items():
        changed = b.state_dict()[name]
        if 'bias_ih_l0' in name or 'bias_hh_l0' in name:
            h = len(original)//3
            assert torch.equal(original[:h], changed[:h])
            assert torch.equal(original[2*h:], changed[2*h:])
            assert torch.all(changed[h:2*h] == (3 if 'bias_ih' in name else 0))
        else:
            assert torch.equal(original, changed)
    gru = b.memory
    inp = torch.zeros(1, 1, gru.input_size)
    out = torch.zeros(1, 1, gru.hidden_size)
    gates = update_gates(gru, inp, out)
    assert torch.allclose(gates, torch.full_like(gates, torch.sigmoid(torch.tensor(3.)).item()))


def test_gate_uses_factors_not_tiny_cross_cells():
    cfg = config()
    result = {'accuracy': .98, 'strata': {key: {'one': {'n': 2048, 'accuracy': .96}} for key in cfg['validation']['gate_factors']}}
    assert gate(result, cfg)
    result['strata']['context']['one']['accuracy'] = .94
    assert not gate(result, cfg)


def test_checkpoint_restores_matching_weights_optimizer_and_rng(tmp_path):
    cfg = config()
    seed_all(10)
    model = MemoryControl(model_config(cfg))
    optimizer = optimizer_for(model, cfg)
    rng = np.random.default_rng(42)
    pool = np.array([np.arange(7), np.arange(7)[::-1]])
    batch = generate(pool, 4, rng, cfg, cfg['stages'][0])
    train_step(model, optimizer, batch, cfg)
    path = tmp_path / 'advance.pt'
    save_checkpoint(path, model, optimizer, rng, cfg, 'A', 1, 500, 500, {}, 2)
    weights = copy.deepcopy(model.state_dict())
    optstate = copy.deepcopy(optimizer.state_dict())
    next_random = rng.random()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    optimizer.param_groups[0]['lr'] = .2
    for entry in optimizer.state.values():
        entry['exp_avg'].zero_()
    rng.random(17)
    state = restore_checkpoint(path, model, optimizer, rng)
    assert state['consecutive_passes'] == 2 and state['stage'] == 1
    assert all(torch.equal(model.state_dict()[k], value) for k, value in weights.items())
    assert optimizer.param_groups[0]['lr'] == .001
    assert all(torch.equal(optimizer.state_dict()['state'][k]['exp_avg'], value['exp_avg']) for k, value in optstate['state'].items())
    assert rng.random() == next_random
