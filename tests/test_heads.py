"""NumPy heads: gradient check, learning on synthetic data, save/load, sigma positivity."""
import numpy as np
import pytest

from rscrank import heads


def test_mlp_gradient_check():
    rng = np.random.default_rng(0)
    m = heads.MLP(5, 7, seed=1)
    X = rng.normal(size=(4, 5))
    y = rng.normal(size=4)

    def loss(params):
        m.set_params(params)
        pred = m.forward(X)
        return float(0.5 * ((pred - y) ** 2).sum())

    pred, cache = m.forward(X, cache=True)
    grads = m.backward(cache, pred - y)
    base = {k: v.copy() for k, v in m.params().items()}
    for k in ("W1", "b1", "w2", "b2"):
        g_num = np.zeros_like(base[k], dtype=float)
        it = np.nditer(base[k], flags=["multi_index"]) if base[k].ndim else [None]
        idxs = [it.multi_index for _ in it] if base[k].ndim else [()]
        for idx in idxs:
            p1 = {kk: vv.copy() for kk, vv in base.items()}
            p2 = {kk: vv.copy() for kk, vv in base.items()}
            if idx == ():
                p1[k] = p1[k] + 1e-6
                p2[k] = p2[k] - 1e-6
            else:
                p1[k][idx] += 1e-6
                p2[k][idx] -= 1e-6
            d = (loss(p1) - loss(p2)) / 2e-6
            if idx == ():
                g_num = np.array(d)
            else:
                g_num[idx] = d
        assert np.allclose(g_num, grads[k], atol=1e-5, rtol=1e-4), k


def test_mean_head_learns_and_roundtrips(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 10))
    y = 2.0 * X[:, 0] - X[:, 1] ** 2 + 0.1 * rng.normal(size=3000)
    h = heads.MeanHead(10, hidden=32, seed=0)
    info = h.fit(X[:2500], y[:2500], X[2500:], y[2500:], lr=3e-3, epochs=30, patience=5, batch_size=128)
    pred = h.predict(X[2500:])
    mse = float(np.mean((pred - y[2500:]) ** 2))
    assert mse < 0.5 * float(np.var(y[2500:]))
    h.save(tmp_path / "mu.npz")
    h2 = heads.MeanHead.load(tmp_path / "mu.npz")
    assert np.allclose(h2.predict(X[:5]), pred[:0].tolist() + h.predict(X[:5]).tolist())


def test_scale_head_positive_and_fits_log_target():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(2000, 6))
    rho = np.exp(0.5 * Z[:, 0]) + 0.05
    s = heads.ScaleHead(6, hidden=16, seed=0)
    s.fit(Z[:1500], rho[:1500], Z[1500:], rho[1500:], lr=3e-3, epochs=40, patience=6, batch_size=128)
    pred = s.predict(Z[1500:])
    assert np.all(pred > heads.EPS)
    err = np.mean((np.log(pred) - np.log(rho[1500:])) ** 2)
    assert err < 0.15


def test_budget_head_and_features():
    rng = np.random.default_rng(3)
    cheap = rng.normal(size=100)
    z = heads.query_level_features(cheap, cheap + 0.1, rng.integers(20, 300, 100), 12.0, 10)
    assert z.shape[0] == 22 and np.all(np.isfinite(z))
    X = heads.pair_features(rng.normal(size=(100, 384)), cheap, rng.normal(size=100), np.arange(1, 101), 12.0, rng.integers(20, 300, 100), 100)
    assert X.shape == (100, 389)
    Z = rng.normal(size=(800, 22))
    logH = 1.0 + 0.8 * Z[:, 3] + 0.05 * rng.normal(size=800)
    b = heads.BudgetHead(22, hidden=16)
    b.fit(Z[:600], logH[:600], Z[600:], logH[600:], lr=3e-3, epochs=40, patience=6, batch_size=64)
    assert np.mean((b.predict(Z[600:]) - logH[600:]) ** 2) < 0.1
