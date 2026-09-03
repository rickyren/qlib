import numpy as np

from qlib.contrib.strategy.optimizer import EnhancedIndexingOptimizer


def test_enhanced_indexing_optimizer_has_its_required_solver() -> None:
    current = np.full(3, 1.0 / 3.0)
    optimized = EnhancedIndexingOptimizer(
        lamb=1.0,
        delta=0.5,
        b_dev=0.2,
        scale_return=False,
    )(
        r=np.array([0.10, 0.00, -0.10]),
        F=np.array([[1.0], [0.0], [-1.0]]),
        cov_b=np.array([[0.01]]),
        var_u=np.full(3, 0.01),
        w0=current,
        wb=current,
    )

    assert np.isclose(optimized.sum(), 1.0)
    assert np.all(optimized >= 0.0)
    assert not np.allclose(optimized, current)
