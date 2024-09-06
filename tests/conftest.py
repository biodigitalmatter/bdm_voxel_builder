import numpy as np
import numpy.typing as npt
import pyopenvdb as vdb
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
    return 100 * random_generator.random(shape) - 100 / 2


@pytest.fixture
def random_pts():
    return _random_pts


def _activate_random_voxels(
    vdb: vdb.GridBase, random_generator: np.random.Generator, value: float = 1.0
) -> tuple[npt.NDArray, int]:
    """Activate random voxels in the grid, return their indices."""

    to_activate = random_generator.integers(0, 100)

    random_indices = random_generator.integers(0, high=300, size=(to_activate, 3))

    accessor = vdb.getAccessor()

    for index in random_indices:
        accessor.setValueOn(index, value)

    return random_indices


@pytest.fixture
def activate_random_voxels():
    return _activate_random_voxels
