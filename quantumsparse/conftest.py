import pytest

parametrize_method      = pytest.mark.parametrize("method", ["R", "U"])
parametrize_interaction = pytest.mark.parametrize("interaction", ["heisenberg", "DM", "biquadratic", "anisotropy", "rhombicity"])
parametrize_N           = pytest.mark.parametrize("N", [2, 3, 4])
parametrize_S           = pytest.mark.parametrize("S", [0.5, 1.0, 1.5, 2.0])

TOLERANCE = 1e-10