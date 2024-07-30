import compas.geometry as cg
import numpy as np
import pyopenvdb as vdb
import pytest

from bdm_voxel_builder.helpers.vdb import _get_vdb_matrix, xform_to_compas, xform_to_vdb


@pytest.fixture
def identity_xform_vdb():
    return vdb.createLinearTransform()


@pytest.fixture
def identity_xform_compas():
    return cg.Transformation()


@pytest.fixture
def other_xform_matrix():
    return [[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]]


@pytest.fixture
def other_xform_vdb(other_xform_matrix):
    return vdb.createLinearTransform(np.array(other_xform_matrix).transpose().tolist())


@pytest.fixture
def other_xform_compas(other_xform_matrix):
    return cg.Transformation.from_matrix(other_xform_matrix)


def assert_equal_vdb_compas_xforms(
    vdb_xform: vdb.Transform, compas_xform: cg.Transformation
):
    assert vdb_xform.isLinear
    vdb_matrix = _get_vdb_matrix(vdb_xform)
    assert vdb_matrix == compas_xform.matrix


def test__get_vdb_matrix(
    identity_xform_vdb, identity_xform_compas, other_xform_vdb, other_xform_compas
):
    assert_equal_vdb_compas_xforms(
        vdb_xform=identity_xform_vdb, compas_xform=identity_xform_compas
    )

    assert_equal_vdb_compas_xforms(
        vdb_xform=other_xform_vdb,
        compas_xform=other_xform_compas,
    )


def test_xform_to_vdb(identity_xform_compas, other_xform_compas):
    vdb_xform = xform_to_vdb(identity_xform_compas)
    assert_equal_vdb_compas_xforms(vdb_xform, identity_xform_compas)

    vdb_xform = xform_to_vdb(other_xform_compas)
    assert_equal_vdb_compas_xforms(vdb_xform, other_xform_compas)


def test_xform_to_compas(identity_xform_vdb, other_xform_vdb):
    compas_xform = xform_to_compas(identity_xform_vdb)
    assert_equal_vdb_compas_xforms(
        compas_xform=compas_xform, vdb_xform=identity_xform_vdb
    )

    compas_xform = xform_to_compas(other_xform_vdb)
    assert_equal_vdb_compas_xforms(compas_xform=compas_xform, vdb_xform=other_xform_vdb)
