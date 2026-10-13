from __future__ import annotations

import copy

import numpy as np
import torch
from torch.nn import functional as F

from ..task import tensor_inputs


def event_indices(x):
    events = {'stimulus_end': [], 'swap_time': [], 'query_end': []}
    for slot in range(4):
        events[f'rule_slot_{slot+1}_end'] = []
    for row, length in enumerate(x['lengths']):
        phase = x['phase'][row, :length].argmax(-1)
        stim = np.flatnonzero(phase == 1)
        rules = np.flatnonzero(phase == 3)
        query = np.flatnonzero(phase == 4)
        assert len(rules) == 4
        # Each slot ends just before the next rule, minus the extra inter-prefix delay.
        # Find blank computation length from the first rule-to-second-rule gap.
        blank = int(rules[1]-rules[0]-1)
        for slot in range(4):
            events[f'rule_slot_{slot+1}_end'].append(int(rules[slot]+blank))
        events['stimulus_end'].append(int(stim[-1]))
        events['swap_time'].append(int(rules[2]-1))
        events['query_end'].append(int(query[-1]))
    return {k: np.asarray(v) for k, v in events.items()}


def fused_trace(model, x):
    c, _ = model.control(x['ctrl'])
    m, _ = model.memory(torch.cat((x['stim'], c, x['phase']), -1))
    row = torch.arange(len(m), device=m.device)
    end = x['lengths']-1
    return model.readout(torch.cat((m[row, end], c[row, end], x['query']), -1)), m, c


def eager_forward(model, x, retain=False):
    """Same learned nn.GRU modules stepwise; carried nodes stay in the graph."""
    batch, steps, _ = x['stim'].shape
    m = x['stim'].new_zeros((batch, model.memory_dim))
    c = x['ctrl'].new_zeros((batch, model.control_dim))
    memories, controls = [], []
    for t in range(steps):
        _, next_c = model.control(x['ctrl'][:, t:t+1], c.unsqueeze(0).contiguous())
        c = next_c.squeeze(0)
        _, next_m = model.memory(torch.cat((x['stim'][:, t], c, x['phase'][:, t]), -1).unsqueeze(1),
                                 m.unsqueeze(0).contiguous())
        m = next_m.squeeze(0)
        if retain:
            m.retain_grad()
            c.retain_grad()
        memories.append(m)
        controls.append(c)
    mm, cc = torch.stack(memories, 1), torch.stack(controls, 1)
    row = torch.arange(batch, device=mm.device)
    end = x['lengths']-1
    logits = model.readout(torch.cat((mm[row, end], cc[row, end], x['query']), -1))
    return logits, memories, controls


def stats(values):
    v = np.asarray(values, np.float64).ravel()
    return {'n': len(v), 'mean': float(v.mean()), 'min': float(v.min()), 'max': float(v.max()),
            'p05': float(np.quantile(v, .05)), 'median': float(np.median(v)), 'p95': float(np.quantile(v, .95))}


def gradient_path(model, data, trace_path=None):
    dev = next(model.parameters()).device
    model.train()  # cuDNN backward requires training reserve; there is no dropout.
    x = tensor_inputs(data, dev)
    x['query'] = torch.as_tensor(data['query'], device=dev)
    x['stim'] = x['stim'].clone().requires_grad_(True)
    target = torch.as_tensor(data['target'], device=dev)
    model.zero_grad(set_to_none=True)
    expected = model(x, x['query'])
    actual_parameters = list(model.named_parameters())
    actual_gradients = torch.autograd.grad(F.cross_entropy(expected, target),
                                         [x['stim'], *[p for name, p in actual_parameters]], allow_unused=True)
    logits, memories, controls = eager_forward(model, x, retain=True)
    discrepancy = float((logits-expected).abs().max().detach())
    assert torch.allclose(logits, expected, atol=3e-6, rtol=3e-5), f'Actual forward differs from eager carry {discrepancy}'
    loss = F.cross_entropy(logits, target)
    loss.backward()
    actual_stim = actual_gradients[0]
    gradient_differences = []
    for (name, parameter), actual in zip(actual_parameters, actual_gradients[1:]):
        assert actual is not None and parameter.grad is not None, f'Actual training graph excluded {name}'
        assert torch.allclose(actual, parameter.grad, atol=2e-6, rtol=2e-3), f'Actual/eager parameter derivative differs: {name}'
        gradient_differences.append(float((actual-parameter.grad).abs().max()))
    assert actual_stim is not None, 'Actual training loss has no stimulus path'
    assert torch.allclose(actual_stim, x['stim'].grad, atol=2e-6, rtol=2e-3)
    records = {}
    for event, indices in event_indices(data).items():
        values, gradients, none = [], [], 0
        for row, t in enumerate(indices):
            values.append(float(memories[t][row].detach().norm()))
            if memories[t].grad is None:
                none += 1
            else:
                gradients.append(float(memories[t].grad[row].norm()))
        records[event] = {'state_l2': stats(values), 'gradient_none_rows': none,
                          'gradient_finite': bool(np.isfinite(gradients).all()),
                          'gradient_l2': stats(gradients) if gradients else None}
    parameters = {}
    for component in ('memory', 'control', 'readout'):
        entries = list(getattr(model, component).named_parameters())
        present = [p.grad for name, p in entries if p.grad is not None]
        parameters[component] = {'none': [name for name, p in entries if p.grad is None],
                                 'finite': all(bool(torch.isfinite(v).all()) for v in present),
                                 'l2': float(torch.sqrt(sum(g.square().sum() for g in present))) if present else None}
    stimgrad = x['stim'].grad
    mask = torch.as_tensor(data['phase'][:, :, 1] == 1, device=dev)
    result = {'batch_size': len(data['target']), 'loss': float(loss.detach()),
              'fused_eager_max_logit_difference': discrepancy, 'events': records, 'parameters': parameters,
              'actual_fused_stimulus_gradient_none': actual_stim is None,
              'actual_fused_stimulus_gradient_finite': bool(torch.isfinite(actual_stim).all()),
              'actual_fused_stimulus_gradient_l2_during_stim': stats(actual_stim.norm(dim=-1)[mask].detach().cpu().numpy()),
              'actual_fused_eager_max_parameter_gradient_difference': max(gradient_differences),
              'stimulus_gradient_none': stimgrad is None,
              'stimulus_gradient_finite': bool(torch.isfinite(stimgrad).all()) if stimgrad is not None else None,
              'stimulus_gradient_l2_during_stim': stats(stimgrad.norm(dim=-1)[mask].detach().cpu().numpy()) if stimgrad is not None else None}
    if trace_path is not None:
        event_map = event_indices(data)
        names = list(event_map)
        states, state_grad, cs, cg = [], [], [], []
        for name in names:
            states.append(torch.stack([memories[t][row].detach() for row, t in enumerate(event_map[name])]).cpu().numpy())
            state_grad.append(torch.stack([memories[t].grad[row] for row, t in enumerate(event_map[name])]).cpu().numpy())
            cs.append(torch.stack([controls[t][row].detach() for row, t in enumerate(event_map[name])]).cpu().numpy())
            cg.append(torch.stack([controls[t].grad[row] for row, t in enumerate(event_map[name])]).cpu().numpy())
        np.savez_compressed(trace_path, event_names=np.array(names), event_indices=np.array([event_map[n] for n in names]),
                            memory_states=np.array(states), memory_gradients=np.array(state_grad),
                            control_states=np.array(cs), control_gradients=np.array(cg),
                            stimulus_input=data['stim'], control_input=data['ctrl'], phase=data['phase'],
                            query=data['query'], lengths=data['lengths'], target=data['target'],
                            actual_fused_stimulus_gradient=actual_stim.detach().cpu().numpy(),
                            eager_stimulus_gradient=stimgrad.detach().cpu().numpy())
    model.eval()
    return result


def finite_difference(model, data):
    # CPU float64 audit isolates derivative correctness from long-delay precision.
    candidate = copy.deepcopy(model).cpu().double().eval()
    x = {k: v.to(dtype=torch.float64) if v.is_floating_point() else v for k, v in tensor_inputs(data, torch.device('cpu')).items()}
    x['query'] = torch.tensor(data['query'], dtype=torch.float64)
    x['stim'].requires_grad_(True)
    t = int(np.flatnonzero(data['phase'][0, :, 1])[0])
    feature = int(np.flatnonzero(data['stim'][0, t])[0])
    output = int(data['target'][0])
    value = candidate(x, x['query'])[0, output]
    derivative = torch.autograd.grad(value, x['stim'])[0][0, t, feature].item()
    epsilon = 1e-5
    with torch.no_grad():
        positive, negative = x['stim'].detach().clone(), x['stim'].detach().clone()
        positive[0, t, feature] += epsilon
        negative[0, t, feature] -= epsilon
        plus = candidate({**x, 'stim': positive}, x['query'])[0, output].item()
        minus = candidate({**x, 'stim': negative}, x['query'])[0, output].item()
    fd = (plus-minus)/(2*epsilon)
    passed = bool(abs(fd-derivative) <= 1e-8 + 1e-5*abs(derivative))
    assert passed, 'Short float64 derivative audit failed'
    return {'dtype': 'float64', 'timestep_zero_based': t, 'stimulus_feature': feature, 'output_class': output,
            'epsilon': epsilon, 'autograd': derivative, 'finite_difference': fd,
            'absolute_error': abs(fd-derivative), 'passed': passed}


def update_gates(gru, inputs, outputs):
    previous = torch.cat((torch.zeros_like(outputs[:, :1]), outputs[:, :-1]), 1)
    ih = F.linear(inputs, gru.weight_ih_l0, gru.bias_ih_l0)
    hh = F.linear(previous, gru.weight_hh_l0, gru.bias_hh_l0)
    h = gru.hidden_size
    return torch.sigmoid(ih[..., h:2*h]+hh[..., h:2*h])


@torch.no_grad()
def gate_and_sensitivity(model, data):
    model.eval()
    dev = next(model.parameters()).device
    x = tensor_inputs(data, dev)
    x['query'] = torch.as_tensor(data['query'], device=dev)
    logits, memory, control = fused_trace(model, x)
    zero_x = {**x, 'stim': torch.zeros_like(x['stim'])}
    _, zero_m, _ = fused_trace(model, zero_x)
    sensitivity = {}
    rows = torch.arange(len(memory), device=dev)
    for name, indices in event_indices(data).items():
        if name in ('stimulus_end', 'swap_time', 'query_end'):
            t = torch.as_tensor(indices, device=dev)
            sensitivity[name] = stats((memory[rows, t]-zero_m[rows, t]).norm(dim=-1).cpu().numpy())
    gates = {'control': update_gates(model.control, x['ctrl'], control),
             'memory': update_gates(model.memory, torch.cat((x['stim'], control, x['phase']), -1), memory)}
    valid = torch.arange(memory.shape[1], device=dev)[None] < x['lengths'][:, None]
    summaries = {}
    for component, array in gates.items():
        summaries[component] = {}
        for phase, name in ((1, 'stimulus'), (3, 'rule'), (2, 'delay'), (4, 'query')):
            mask = valid & (x['phase'][:, :, phase] == 1)
            if mask.any():
                values = array[mask].cpu().numpy()
                summaries[component][name] = {**stats(values), 'fraction_above_0_9': float(np.mean(values > .9))}
            else:
                summaries[component][name] = None
    return {'content_sensitivity': sensitivity, 'update_gates': summaries}
