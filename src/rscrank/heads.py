"""Small trainable heads on top of frozen cheap features (Sec. 10.3-10.6).

Implemented in NumPy (Adam, mini-batches, early stopping) so that the whole
pipeline can be smoke-tested without a GPU framework; the feature dimension is
~390 and the training set ~1e6 pairs, which NumPy handles in minutes.

* MeanHead   : mu_i = w2^T GELU(W1 x_i + b1) + b2, Huber loss (eq. 25-26)
* ScaleHead  : sigma(q) = eps + softplus(g(z_q)), loss (log sigma - log rho)^2 (eq. 27-28)
* BudgetHead : h(q) ~ log H_k(q)  (query-adaptive prefix baseline, Sec. 14.3)

All standardisation statistics come from the training split only.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["MLP", "MeanHead", "ScaleHead", "BudgetHead", "query_level_features", "pair_features", "rho_target"]

EPS = 1e-3


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    t = np.tanh(0.7978845608028654 * (x + 0.044715 * x**3))
    dt = (1 - t * t) * 0.7978845608028654 * (1 + 3 * 0.044715 * x * x)
    return 0.5 * (1 + t) + 0.5 * x * dt


def _huber(e: np.ndarray, kappa: float) -> Tuple[np.ndarray, np.ndarray]:
    a = np.abs(e)
    loss = np.where(a <= kappa, 0.5 * e * e, kappa * (a - 0.5 * kappa))
    grad = np.where(a <= kappa, e, kappa * np.sign(e))
    return loss, grad


@dataclass
class MLP:
    """Two-layer MLP with GELU; trained by Adam on a user-supplied loss gradient."""
    d_in: int
    d_hidden: int = 128
    seed: int = 0
    W1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    w2: np.ndarray = field(init=False)
    b2: float = field(init=False)
    x_mean: Optional[np.ndarray] = None
    x_std: Optional[np.ndarray] = None

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(0, math.sqrt(2.0 / self.d_in), (self.d_in, self.d_hidden))
        self.b1 = np.zeros(self.d_hidden)
        self.w2 = rng.normal(0, math.sqrt(1.0 / self.d_hidden), self.d_hidden)
        self.b2 = 0.0

    def fit_standardiser(self, X: np.ndarray) -> None:
        self.x_mean = X.mean(0)
        self.x_std = X.std(0) + 1e-6

    def _z(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / self.x_std if self.x_mean is not None else X

    def forward(self, X: np.ndarray, cache: bool = False):
        Z = self._z(np.asarray(X, dtype=np.float64))
        A = Z @ self.W1 + self.b1
        Hh = _gelu(A)
        out = Hh @ self.w2 + self.b2
        if cache:
            return out, (Z, A, Hh)
        return out

    def backward(self, cache, dout: np.ndarray) -> Dict[str, np.ndarray]:
        Z, A, Hh = cache
        g_w2 = Hh.T @ dout
        g_b2 = float(dout.sum())
        dH = np.outer(dout, self.w2) * _gelu_grad(A)
        g_W1 = Z.T @ dH
        g_b1 = dH.sum(0)
        return {"W1": g_W1, "b1": g_b1, "w2": g_w2, "b2": g_b2}

    def params(self) -> Dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "w2": self.w2, "b2": np.array(self.b2)}

    def set_params(self, p: Dict[str, np.ndarray]) -> None:
        self.W1, self.b1, self.w2, self.b2 = p["W1"].copy(), p["b1"].copy(), p["w2"].copy(), float(p["b2"])

    def save(self, path: Path) -> None:
        np.savez(path, W1=self.W1, b1=self.b1, w2=self.w2, b2=self.b2, x_mean=self.x_mean, x_std=self.x_std, d_in=self.d_in, d_hidden=self.d_hidden)

    @classmethod
    def load(cls, path: Path) -> "MLP":
        z = np.load(path, allow_pickle=False)
        m = cls(int(z["d_in"]), int(z["d_hidden"]))
        m.W1, m.b1, m.w2, m.b2 = z["W1"], z["b1"], z["w2"], float(z["b2"])
        m.x_mean, m.x_std = z["x_mean"], z["x_std"]
        return m


def train_mlp(model: MLP, X: np.ndarray, y: np.ndarray, loss_grad, X_dev: Optional[np.ndarray] = None, y_dev: Optional[np.ndarray] = None, sample_weight: Optional[np.ndarray] = None, lr: float = 1e-3, weight_decay: float = 0.0, epochs: int = 20, patience: int = 3, batch_size: int = 512, seed: int = 0, log: Optional[List[Dict]] = None) -> Dict[str, float]:
    """Generic Adam loop.  ``loss_grad(pred, y)`` returns (loss_vector, grad_vector)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    w = w / w.mean()
    m = {k: np.zeros_like(v, dtype=float) for k, v in model.params().items()}
    v = {k: np.zeros_like(val, dtype=float) for k, val in model.params().items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    best, best_params, bad = math.inf, model.params(), 0
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            pred, cache = model.forward(X[idx], cache=True)
            loss, g = loss_grad(pred, y[idx])
            tot += float((loss * w[idx]).sum())
            grads = model.backward(cache, g * w[idx] / len(idx))
            step += 1
            p = model.params()
            new = {}
            for k in p:
                gk = grads[k] + weight_decay * p[k] if k in ("W1", "w2") else grads[k]
                m[k] = b1 * m[k] + (1 - b1) * gk
                v[k] = b2 * v[k] + (1 - b2) * (gk * gk)
                mh = m[k] / (1 - b1**step)
                vh = v[k] / (1 - b2**step)
                new[k] = p[k] - lr * mh / (np.sqrt(vh) + eps)
            model.set_params(new)
        train_loss = tot / n
        if X_dev is not None:
            dl, _ = loss_grad(model.forward(X_dev), y_dev)
            dev_loss = float(dl.mean())
        else:
            dev_loss = train_loss
        if log is not None:
            log.append({"epoch": ep, "train_loss": train_loss, "dev_loss": dev_loss})
        if dev_loss < best - 1e-9:
            best, best_params, bad = dev_loss, {k: val.copy() for k, val in model.params().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.set_params(best_params)
    return {"best_dev_loss": best, "epochs_run": ep + 1}


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #
def pair_features(emb: np.ndarray, cheap_raw: np.ndarray, bm25: np.ndarray, bm25_rank: np.ndarray, q_len: float, d_len: np.ndarray, N: int) -> np.ndarray:
    """x_i = [h_i ; a_i ; v_i ; p_i/N ; log(1+|q|) ; log(1+|d_i|)]  (Sec. 10.3)."""
    n = len(cheap_raw)
    extra = np.stack([cheap_raw, bm25, bm25_rank / float(N), np.full(n, math.log1p(q_len)), np.log1p(d_len)], axis=1)
    return np.concatenate([np.asarray(emb, dtype=np.float64), extra], axis=1)


def query_level_features(cheap_raw: np.ndarray, mu: Optional[np.ndarray], d_len: np.ndarray, q_len: float, k: int) -> np.ndarray:
    """z(q): cheap-score statistics, cheap-rank boundary gaps and length statistics (Sec. 10.5).
    Uses only cheap information (never teacher scores)."""
    a = np.sort(np.asarray(cheap_raw, dtype=float))[::-1]
    feats = [a.mean(), a.std(), a[0], a[min(k - 1, len(a) - 1)], a[min(k, len(a) - 1)], a[-1], a[min(k - 1, len(a) - 1)] - a[min(k, len(a) - 1)], a[0] - a[min(k, len(a) - 1)]]
    feats += list(np.quantile(a, [0.1, 0.25, 0.5, 0.75, 0.9]))
    if mu is not None:
        m = np.sort(np.asarray(mu, dtype=float))[::-1]
        feats += [m.mean(), m.std(), m[min(k - 1, len(m) - 1)] - m[min(k, len(m) - 1)], m[0] - m[min(k, len(m) - 1)]]
    else:
        feats += [0.0, 0.0, 0.0, 0.0]
    dl = np.log1p(np.asarray(d_len, dtype=float))
    # list length: constant (= N) in the main configuration, so the standardiser makes it degenerate; when frozen heads
    # trained at one N are applied to a longer list (N18, N = 200), RSC_Z_LEN_OVERRIDE clamps it to the training value
    n_feat = float(os.environ.get("RSC_Z_LEN_OVERRIDE") or len(a))
    feats += [dl.mean(), dl.std(), dl.max(), math.log1p(q_len), n_feat]
    return np.asarray(feats, dtype=np.float64)


def rho_target(s: np.ndarray, mu: np.ndarray, top_positions: np.ndarray) -> float:
    """rho(q) = eps + sqrt( mean_{i in T_k} (s_i - mu_i)^2 )."""
    e = s[top_positions] - mu[top_positions]
    return EPS + float(np.sqrt(np.mean(e * e)))


# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #
class MeanHead:
    def __init__(self, d_in: int, hidden: int = 128, kappa: float = 1.0, seed: int = 0):
        self.mlp = MLP(d_in, hidden, seed)
        self.kappa = kappa

    def fit(self, X, y, X_dev=None, y_dev=None, sample_weight=None, **kw) -> Dict[str, float]:
        self.mlp.fit_standardiser(X)
        lg = lambda pred, yy: _huber(pred - yy, self.kappa)
        return train_mlp(self.mlp, X, y, lg, X_dev, y_dev, sample_weight, **kw)

    def predict(self, X) -> np.ndarray:
        return self.mlp.forward(X)

    def save(self, path: Path):
        self.mlp.save(path)

    @classmethod
    def load(cls, path: Path, kappa: float = 1.0) -> "MeanHead":
        obj = cls.__new__(cls)
        obj.mlp = MLP.load(path)
        obj.kappa = kappa
        return obj


class ScaleHead:
    """sigma(q) = EPS + softplus(g(z)); trained on log-targets."""

    def __init__(self, d_in: int, hidden: int = 32, seed: int = 0):
        self.mlp = MLP(d_in, hidden, seed)

    @staticmethod
    def _sigma(g: np.ndarray) -> np.ndarray:
        return EPS + np.logaddexp(0.0, g)

    def fit(self, Z, rho, Z_dev=None, rho_dev=None, **kw) -> Dict[str, float]:
        self.mlp.fit_standardiser(Z)
        lr_ = np.log(np.asarray(rho, dtype=float))
        lr_dev = np.log(np.asarray(rho_dev, dtype=float)) if rho_dev is not None else None

        def lg(g, target):
            sig = self._sigma(g)
            e = np.log(sig) - target
            dsig_dg = 1.0 / (1.0 + np.exp(-g))  # softplus'
            grad = 2 * e * (dsig_dg / sig)
            return e * e, grad

        return train_mlp(self.mlp, Z, lr_, lg, Z_dev, lr_dev, **kw)

    def predict(self, Z) -> np.ndarray:
        return self._sigma(self.mlp.forward(Z))

    def save(self, path: Path):
        self.mlp.save(path)

    @classmethod
    def load(cls, path: Path) -> "ScaleHead":
        obj = cls.__new__(cls)
        obj.mlp = MLP.load(path)
        return obj


class BudgetHead:
    """h(q) ~ log H_k(q), squared loss."""

    def __init__(self, d_in: int, hidden: int = 32, seed: int = 0):
        self.mlp = MLP(d_in, hidden, seed)

    def fit(self, Z, logH, Z_dev=None, logH_dev=None, **kw) -> Dict[str, float]:
        self.mlp.fit_standardiser(Z)
        lg = lambda pred, yy: ((pred - yy) ** 2, 2 * (pred - yy))
        return train_mlp(self.mlp, Z, np.asarray(logH, float), lg, Z_dev, None if logH_dev is None else np.asarray(logH_dev, float), **kw)

    def predict(self, Z) -> np.ndarray:
        return self.mlp.forward(Z)

    def save(self, path: Path):
        self.mlp.save(path)

    @classmethod
    def load(cls, path: Path) -> "BudgetHead":
        obj = cls.__new__(cls)
        obj.mlp = MLP.load(path)
        return obj
