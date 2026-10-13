from __future__ import annotations

import numpy as np
import torch

from .task import prefix_inputs, tensor_inputs
from .truth import CONDITIONS, apply, matrix_apply, preimage


@torch.no_grad()
def cache_states(model, q, orbit_ids, cfg, out=None):
    dev = next(model.parameters()).device
    model.eval()
    n = cfg['n']
    memory, control, initials, trajectories = [], [], [], []
    for d in cfg['timing']['delays']:
        mm, cc, pp, tt = [], [], [], []
        for r, history in CONDITIONS:
            p = preimage(q, history, r)
            assert np.array_equal(matrix_apply(p, history, r), q), 'Independent content match failed'
            traj = np.stack((p, apply(p, history[0], r), apply(p, history, r)), axis=1)
            assert np.array_equal(traj[:, -1], q)
            batch_m, batch_c = [], []
            size = cfg['execution']['cache_batch_size']
            for start in range(0, len(q), size):
                ppart = p[start:start + size]
                x = prefix_inputs(ppart, np.full(len(ppart), r), [history] * len(ppart),
                                  np.full(len(ppart), d), cfg)
                m, c = model.states(tensor_inputs(x, dev))
                batch_m.append(m.cpu().numpy())
                batch_c.append(c.cpu().numpy())
            m = np.concatenate(batch_m)
            c = np.concatenate(batch_c)
            assert np.max(np.abs(c - c[0])) <= 1e-6, 'Control leaked content or cache error'
            mm.append(m)
            cc.append(c[0])
            pp.append(p)
            tt.append(traj)
        memory.append(mm)
        control.append(cc)
        initials.append(pp)
        trajectories.append(tt)
    result = {'memory': np.asarray(memory), 'control': np.asarray(control),
              'initial': np.asarray(initials), 'truth_trajectory': np.asarray(trajectories),
              'content': np.asarray(q), 'orbit_id': np.asarray(orbit_ids),
              'delays': np.asarray(cfg['timing']['delays']),
              'contexts': np.array([r for r, h in CONDITIONS]),
              'histories': np.array([h for r, h in CONDITIONS]),
              'cache_steps': np.array([cfg['timing']['context'] + cfg['timing']['stimulus'] +
                                      cfg['timing']['initial_delay'] +
                                      2*(cfg['timing']['rule_cue'] + cfg['timing']['rule_blank']) + d
                                      for d in cfg['timing']['delays']])}
    if out is not None:
        np.savez_compressed(out, **result)
    return result
