"""Neural estimators as sklearn-compatible wrappers.

The GCN uses a dense normalised adjacency over the feature nodes it is
actually trained on. torch_geometric is deliberately not a dependency (see
Global Constraints) - with only tens of nodes a hand-rolled dense-adjacency
layer is a few lines of plain torch.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin

from . import config


def build_adjacency(corr, threshold=0.3):
    """Symmetric, self-looped, degree-normalised adjacency from a correlation matrix."""
    A = np.abs(np.asarray(corr, dtype=float)).copy()
    A[A < threshold] = 0.0
    np.fill_diagonal(A, 1.0)
    A = (A + A.T) / 2.0
    d = A.sum(axis=1)
    d[d == 0] = 1.0
    dinv = np.diag(1.0 / np.sqrt(d))
    return dinv @ A @ dinv


class _TorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden=64, max_epochs=200, lr=1e-3, random_state=config.SEED):
        self.hidden = hidden
        self.max_epochs = max_epochs
        self.lr = lr
        self.random_state = random_state

    def _build(self, n_features):
        raise NotImplementedError

    def _loss(self, X, y):
        """Training objective. Subclasses override to add auxiliary terms."""
        return nn.functional.mse_loss(self._forward(X), y)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        X = torch.tensor(np.asarray(X, dtype=np.float32))
        y = torch.tensor(np.asarray(y, dtype=np.float32)).view(-1, 1)
        self.net_ = self._build(X.shape[1])
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        self.net_.train()
        for _ in range(self.max_epochs):
            opt.zero_grad()
            self._loss(X, y).backward()
            opt.step()
        return self

    def _forward(self, X):
        return self.net_(X)

    def predict(self, X):
        self.net_.eval()
        with torch.no_grad():
            out = self._forward(torch.tensor(np.asarray(X, dtype=np.float32)))
        return out.view(-1).numpy()


class MLPRegressorTorch(_TorchRegressor):
    def _build(self, n_features):
        return nn.Sequential(
            nn.Linear(n_features, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden // 2), nn.ReLU(),
            nn.Linear(self.hidden // 2, 1),
        )


class AutoencoderRegressor(_TorchRegressor):
    """Compress to a latent code, then regress from it. The reconstruction term
    keeps the latent faithful to the input structure."""

    def __init__(self, hidden=64, latent=8, max_epochs=200, lr=1e-3,
                 recon_weight=0.5, random_state=config.SEED):
        super().__init__(hidden, max_epochs, lr, random_state)
        self.latent = latent
        self.recon_weight = recon_weight

    def _build(self, n_features):
        self.encoder_ = nn.Sequential(
            nn.Linear(n_features, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.latent))
        self.decoder_ = nn.Sequential(
            nn.Linear(self.latent, self.hidden), nn.ReLU(), nn.Linear(self.hidden, n_features))
        self.head_ = nn.Sequential(nn.Linear(self.latent, 16), nn.ReLU(), nn.Linear(16, 1))
        return nn.ModuleList([self.encoder_, self.decoder_, self.head_])

    def _forward(self, X):
        return self.head_(self.encoder_(X))

    def _loss(self, X, y):
        z = self.encoder_(X)
        return (nn.functional.mse_loss(self.head_(z), y)
                + self.recon_weight * nn.functional.mse_loss(self.decoder_(z), X))


class GCNRegressor(_TorchRegressor):
    """Two-layer GCN over feature-nodes. Each article is a signal on the graph:
    H = A_hat @ diag(x) is approximated by scaling node embeddings by feature values.

    When `adjacency` is not supplied, the graph is built at fit time from the
    training matrix itself (Pearson correlation of columns, thresholded and
    normalised via `build_adjacency`). This is deliberate, not a convenience
    default: `benchmark.build_xy` returns 43 sub-dimensions plus `year`, in an
    order and membership that differs from `config.SCORE_COLS` even though the
    column count coincidentally also lands on 44. A `SCORE_COLS`-ordered
    adjacency passed in from outside would be silently misaligned with the
    columns it multiplies - wrong in a way that looks correct. Deriving the
    graph from X at fit time guarantees the adjacency always matches the
    features actually being trained on.
    """

    def __init__(self, hidden=32, max_epochs=200, lr=1e-3,
                 adjacency=None, threshold=0.3, random_state=config.SEED):
        super().__init__(hidden, max_epochs, lr, random_state)
        self.adjacency = adjacency
        self.threshold = threshold

    def _build(self, n_features):
        # self.A_ is set in fit() before this is called, so it always matches
        # n_features by construction - no reshape-to-identity fallback needed.
        self.w1_ = nn.Linear(1, self.hidden)
        self.w2_ = nn.Linear(self.hidden, self.hidden)
        self.head_ = nn.Sequential(nn.Linear(n_features * self.hidden, 32), nn.ReLU(),
                                   nn.Linear(32, 1))
        return nn.ModuleList([self.w1_, self.w2_, self.head_])

    def _forward(self, X):
        h = X.unsqueeze(-1)                      # (batch, nodes, 1)
        h = torch.relu(self.A_ @ self.w1_(h))    # (batch, nodes, hidden)
        h = torch.relu(self.A_ @ self.w2_(h))
        return self.head_(h.flatten(start_dim=1))

    def fit(self, X, y):
        Xarr = np.asarray(X, dtype=float)
        if self.adjacency is None:
            corr = np.corrcoef(Xarr.T)
            corr = np.nan_to_num(corr, nan=0.0)   # constant columns give NaN
            A = build_adjacency(corr, threshold=self.threshold)
        else:
            A = np.asarray(self.adjacency, dtype=float)
        self.A_ = torch.tensor(A, dtype=torch.float32)
        return super().fit(X, y)
