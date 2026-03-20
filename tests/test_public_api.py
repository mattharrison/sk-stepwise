import sk_stepwise as sw


def test_public_api_exports_expected_symbols():
    assert sw.Categorical.__module__ == "sk_stepwise.spaces"
    assert sw.Int.__module__ == "sk_stepwise.spaces"
    assert sw.Float.__module__ == "sk_stepwise.spaces"
    assert sw.StepwiseOptunaSearchCV.__module__ == "sk_stepwise.search"
    assert sw.StepwiseHyperoptOptimizer.__module__ == "sk_stepwise.search"
