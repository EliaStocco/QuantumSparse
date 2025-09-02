import numpy as np
from quantumsparse.matrix.matrix import Matrix

def make_matrix():
    """Helper to create a small 3x3 Matrix with some entries."""
    data = np.array([1e-12, 2.0, -3.0])   # one tiny, two significant
    row = np.array([0, 1, 2])
    col = np.array([0, 1, 2])
    return Matrix((data, (row, col)), shape=(3, 3))


def test_clean_removes_small_entries():
    m = make_matrix()
    m.clean(noise=1e-10)

    # Small entry should be gone
    assert m[0, 0] == 0.0
    assert m.nnz == 2
    assert set(m.data) == {2.0, -3.0}


def test_clean_keeps_large_entries():
    m = make_matrix()
    m.clean(noise=1e-20)

    # Nothing removed
    assert m.nnz == 3
    assert np.allclose(m.diagonal(), [1e-12, 2.0, -3.0])


def test_clean_removes_all_if_threshold_high():
    m = make_matrix()
    m.clean(noise=10.0)

    # Entire matrix should be zero
    assert m.nnz == 0
    assert m.shape == (3, 3)


def test_clean_with_empty_matrix():
    m = Matrix(([], ([], [])), shape=(2, 2))
    m.clean(noise=1e-10)

    assert m.nnz == 0
    assert m.shape == (2, 2)


def test_clean_does_not_break_eigenstates(monkeypatch):
    m = make_matrix()

    class DummyEigen:
        called = False
        def clean(self):
            DummyEigen.called = True

    m.eigenstates = DummyEigen()
    m.clean(noise=1e-10)

    assert DummyEigen.called
