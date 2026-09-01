import numpy as np
import pytest
from sklearn.base import BaseEstimator
from analysis import config, descriptives, loading, models_nn, quality

DF = quality.clean(loading.load_scoring())
RNG = np.random.default_rng(config.SEED)

@pytest.mark.parametrize("cls", [models_nn.MLPRegressorTorch,
                                 models_nn.AutoencoderRegressor,
                                 models_nn.GCNRegressor])
def test_estimators_follow_the_sklearn_contract(cls):
    assert issubclass(cls, BaseEstimator)
    est = cls(max_epochs=5, random_state=config.SEED)
    assert hasattr(est, "fit") and hasattr(est, "predict")
    assert est.get_params()["max_epochs"] == 5

@pytest.mark.parametrize("cls", [models_nn.MLPRegressorTorch,
                                 models_nn.AutoencoderRegressor,
                                 models_nn.GCNRegressor])
def test_estimators_fit_and_predict_the_right_shape(cls):
    X = RNG.normal(size=(200, 44)); y = X[:, :3].sum(axis=1) + RNG.normal(0, .1, 200)
    pred = cls(max_epochs=20, random_state=config.SEED).fit(X, y).predict(X)
    assert pred.shape == (200,)
    assert np.isfinite(pred).all()

@pytest.mark.parametrize("cls", [models_nn.MLPRegressorTorch,
                                 models_nn.AutoencoderRegressor,
                                 models_nn.GCNRegressor])
def test_estimators_learn_a_linear_signal(cls):
    X = RNG.normal(size=(600, 44)); y = 2.0 * X[:, 0] + X[:, 1]
    est = cls(max_epochs=150, random_state=config.SEED).fit(X, y)
    mae_model = np.abs(est.predict(X) - y).mean()
    mae_mean = np.abs(y.mean() - y).mean()
    assert mae_model < mae_mean

@pytest.mark.parametrize("cls", [models_nn.MLPRegressorTorch,
                                 models_nn.AutoencoderRegressor,
                                 models_nn.GCNRegressor])
def test_estimators_are_reproducible(cls):
    X = RNG.normal(size=(150, 44)); y = X[:, 0]
    a = cls(max_epochs=15, random_state=config.SEED).fit(X, y).predict(X)
    b = cls(max_epochs=15, random_state=config.SEED).fit(X, y).predict(X)
    assert np.allclose(a, b)

def test_adjacency_is_symmetric_with_self_loops():
    corr = descriptives.correlation_matrix(DF).to_numpy()
    A = models_nn.build_adjacency(corr, threshold=0.3)
    assert A.shape == (44, 44)
    assert np.allclose(A, A.T)
    assert (np.diag(A) > 0).all()

def test_adjacency_thresholding_drops_weak_edges():
    corr = np.array([[1.0, 0.9, 0.05], [0.9, 1.0, 0.02], [0.05, 0.02, 1.0]])
    A = models_nn.build_adjacency(corr, threshold=0.3)
    assert A[0, 1] > 0
    assert A[0, 2] == 0

def test_benchmark_exposes_the_neural_estimators():
    from analysis import benchmark
    assert {"MLP", "Autoencoder", "GNN"} <= set(benchmark.make_estimators(config.SEED))

def test_fitted_gcn_uses_a_real_graph_not_the_identity():
    """A GCN on an identity adjacency is just a per-node MLP. The adjacency must
    be derived from the training features when none is supplied."""
    X = RNG.normal(size=(200, 10))
    X[:, 1] = X[:, 0] + RNG.normal(0, 0.01, 200)      # columns 0 and 1 strongly correlated
    y = X[:, 0]
    est = models_nn.GCNRegressor(max_epochs=5, random_state=config.SEED).fit(X, y)
    A = est.A_.numpy()
    assert A.shape == (10, 10)
    assert not np.allclose(A, np.eye(10)), "adjacency collapsed to the identity"
    assert A[0, 1] > 0, "correlated features must be connected"
