import sk_stepwise as sw
import pytest


def test_initialization():
    model = None
    rounds = []
    optimizer = sw.StepwiseHyperoptOptimizer(model, rounds)
    assert optimizer is not None


@pytest.mark.xfail(raises=TypeError)
def test_logistic():
    from sklearn import linear_model

    model = linear_model.LinearRegression()
    rounds = []
    opt = sw.StepwiseHyperoptOptimizer(model, rounds)
    X = [[0, 1], [0, 2]]
    y = [1, 0]
    opt.fit(X, y)


@pytest.mark.matt
def test_matt():
    assert "matt" == "matt"
