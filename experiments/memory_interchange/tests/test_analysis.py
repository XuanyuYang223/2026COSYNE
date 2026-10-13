import itertools

import numpy as np
import pytest
import torch

from ..common import config
from ..intervene import point_metrics, row_metrics
from ..predict import fit_predict, leave_seed_out
from ..truth import CONDITIONS, SUFFIXES
from ..common import restore_rng, rng_state, seed_all
from ..intervene import continue_pairs
from ..model import MemoryControl
from ..task import suffix_inputs, tensor_inputs


def test_positive_damage_does_not_cancel_and_behavior_is_separate():
    native = np.log(np.array([.8, .2, .2, .8]).reshape(1, 1, 2, 1, 2))
    swapped = np.log(np.array([.2, .8, .8, .2]).reshape(1, 1, 1, 2, 1, 2))
    answers = np.zeros((1, 1, 2, 1), int)
    points = point_metrics(native, swapped, swapped, answers, np.ones_like(answers))
    metrics = row_metrics(points)
    assert np.isclose(metrics['signed_loss'], 0)
    assert np.isclose(metrics['loss_positive'], np.log(4)/2)
    assert metrics['new_error'] == .5
    assert metrics['swap_correct'] == .5
    assert metrics['wrong_cf_correct_different'] == .5


def synthetic_rows():
    rng = np.random.default_rng(81)
    rows = []
    for seed in config()['seeds']:
        for a, b in itertools.combinations(range(18), 2):
            for d in (0, 10):
                for s in SUFFIXES:
                    rows.append({'seed': seed, 'a': a, 'b': b, 'd': d, 'S': s,
                                 'cross_context': int(CONDITIONS[a][0] != CONDITIONS[b][0]),
                                 'unseen_pair': int(CONDITIONS[a][1] in ('AB', 'BA') or CONDITIONS[b][1] in ('AB', 'BA')),
                                 'native_ce': rng.random(), 'probe_item_acc': rng.random(),
                                 'control_distance': rng.random(), 'cka': rng.random(), 'init_cka': rng.random(),
                                 'raw': rng.random(), 'proc': rng.random(), 'loss_positive': rng.random()})
    return rows


def test_holdouts_and_missing_values_are_fitted_only_on_training():
    cfg = config()
    rows = synthetic_rows()
    p1, scores, coefficients = leave_seed_out(rows, cfg)
    assert len(p1) == 6 * 153 * 6
    assert coefficients['101_M0']['fitting_seeds'] == [102, 103, 104, 105, 106]
    changed = [{**r, 'loss_positive': r['loss_positive'] + (10000 if r['unseen_pair'] else 0)} for r in rows]
    p2, _, _ = leave_seed_out(changed, cfg)
    for a, b in zip(p1, p2):
        for name in ('M0', 'M_CKA', 'M_raw', 'M_proc', 'M_init', 'M_init_CKA'):
            assert a[name] == b[name]  # No unseen pair label entered any fitting fold.
    poisoned = [{**r, 'loss_positive': r['loss_positive'] + (10000 if r['seed'] == 101 else 0)} for r in rows]
    p3, _, _ = leave_seed_out(poisoned, cfg)
    for a, b in zip(p1, p3):
        if a['seed'] == 101:
            assert a['M_CKA'] == b['M_CKA']
    _, prep, _ = fit_predict(np.array([[0., np.nan], [2., 1.]]), np.array([0., 1.]),
                             np.array([[1e9, 1e9]]))
    assert prep['median'].tolist() == [1., 1.]
    assert prep['mean'].tolist() == [1., 1.]
    with pytest.raises(ValueError):
        leave_seed_out([r for r in rows if r['seed'] != 106], cfg)


@pytest.mark.parametrize('dev', ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'])
def test_batched_intervention_preserves_donor_receiver_query_axes(dev):
    seed_all(43)
    cfg = config()
    cfg['execution']['evaluation_batch_size'] = 11  # boundaries split q/type/pair
    model = MemoryControl(cfg).to(dev).eval()
    rng = np.random.default_rng(4)
    memory = rng.normal(size=(3, 3, 128)).astype(np.float32)
    control = rng.normal(size=(3, 32)).astype(np.float32)
    donors, receivers = [2, 0], [1, 2]
    actual = continue_pairs(model, memory, control, donors, receivers, 'NB', cfg)
    n = cfg['n']
    future = suffix_inputs(['NB']*(2*n), np.repeat([0, 1], n), np.tile(np.arange(n), 2), cfg)
    x = tensor_inputs(future, torch.device(dev))
    with torch.no_grad():
        for pair, (donor, receiver) in enumerate(zip(donors, receivers)):
            for q in range(3):
                m = torch.tensor(memory[donor, q], device=dev).expand(2*n, -1)
                c = torch.tensor(control[receiver], device=dev).expand(2*n, -1)
                expected = model(x, torch.tensor(future['query'], device=dev), m, c).log_softmax(-1)
                np.testing.assert_allclose(actual[pair, q].reshape(2*n, n), expected.cpu().numpy(), rtol=1e-6, atol=1e-6)


def test_rng_restores_python_numpy_torch_and_generator():
    import random
    seed_all(19)
    generator = np.random.default_rng(91)
    state = rng_state(generator)
    reference = (random.random(), np.random.random(), generator.random(), torch.rand(3))
    restore_rng(state, generator)
    actual = (random.random(), np.random.random(), generator.random(), torch.rand(3))
    assert reference[:3] == actual[:3]
    assert torch.equal(reference[3], actual[3])
