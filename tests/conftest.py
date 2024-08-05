import numpy as np
import numpy.typing as npt
import pytest


@pytest.fixture
def random_generator():
    return np.random.default_rng(1000)


@pytest.fixture
def random_int_array(random_generator):
    shape = (10, 10, 10)
    return random_generator.integers(0, 2, size=shape)


def _random_pts(n_pts: int, random_generator: np.random.Generator):
    shape = (n_pts, 3)
    return 100 * random_generator.random(shape) - 100/2


@pytest.fixture
def random_pts():
    return _random_pts


def _activate_random_indices(
    array: npt.NDArray, random_generator: np.random.Generator
) -> tuple[npt.NDArray, int]:
    """Activate random voxels in the grid, return number activated."""
    random_array = random_generator.integers(0, high=2, size=array.shape).astype(
        array.dtype
    )

    return random_array, len(np.flatnonzero(random_array))


@pytest.fixture
def activate_random_indices():
    return _activate_random_indices
