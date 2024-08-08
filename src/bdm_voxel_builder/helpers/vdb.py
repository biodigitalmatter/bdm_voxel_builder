import compas.geometry as cg
import numpy as np
import pyopenvdb as vdb


def _get_vdb_matrix(vdb_xform: vdb.Transform) -> list[list[float]]:
    """Extracts the matrix from a vdb.Transform object.

    Haven't figured out a better way to do this yet.
    """
    text = vdb_xform.info().strip().split("\n")[2:]
    M = []
    for line in text:
        str_parts = (
            line.replace("[", "").replace("]", "").replace(",", "").strip().split()
        )

        float_parts = [float(s) for s in str_parts]
        M.append(float_parts)
    return np.array(M).transpose().tolist()


def xform_to_vdb(xform: cg.Transformation) -> vdb.Transform:
    return vdb.createLinearTransform(np.array(xform.matrix).transpose())


def xform_to_compas(xform: vdb.Transform) -> cg.Transformation:
    return cg.Transformation.from_matrix(_get_vdb_matrix(xform))
