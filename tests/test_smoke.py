def test_imports():
    """Basic smoke test to ensure modules import correctly."""
    import exp.exp_DS as exp_ds
    assert hasattr(exp_ds, 'NeuralModel')
    assert hasattr(exp_ds, 'trainer')
