import numpy as np

from ..geometry import linear_cka, procrustes_distance, raw_distance, reference_cka, ridge_apply, ridge_fit


def test_geometry_independent_formula_and_invariances():
    rng = np.random.default_rng(17)
    x, y = rng.normal(size=(31, 9)), rng.normal(size=(31, 9))
    value = linear_cka(x, y)
    assert np.isclose(value, reference_cka(x, y), atol=1e-12)
    assert np.isclose(value, linear_cka(x * 7, y * 0.25), atol=1e-12)
    order = rng.permutation(len(x))
    assert np.isclose(value, linear_cka(x[order], y[order]), atol=1e-12)
    q, _ = np.linalg.qr(rng.normal(size=(9, 9)))
    z = 3 * x @ q + 10
    assert np.isclose(linear_cka(x, z), 1.)
    assert procrustes_distance(x, z) < 1e-25
    assert raw_distance(x, z) > 0.1
    assert np.isclose(linear_cka(np.eye(4), np.eye(4)), 1.)
    assert np.isnan(linear_cka(np.ones((5, 3)), np.ones((5, 3))))
    assert np.isnan(procrustes_distance(np.ones((5, 3)), np.ones((5, 3))))
    # Centered orthogonal one-dimensional sample vectors give zero CKA.
    a = np.array([-1, -1, 1, 1])[:, None]
    b = np.array([-1, 1, -1, 1])[:, None]
    assert linear_cka(a, b) == 0


def test_ridge_closed_form_and_train_centering():
    x = np.array([[-1.], [1.]])
    fit = ridge_fit(x, 2 * x + 3, alpha=1)
    assert np.isclose(fit['weights'][0, 0], 4/3)
    assert np.isclose(ridge_apply(np.array([[0.]]), fit)[0, 0], 3)
