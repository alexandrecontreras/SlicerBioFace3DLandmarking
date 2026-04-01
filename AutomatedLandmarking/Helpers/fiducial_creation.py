"""
Convert landmark arrays to Slicer fiducial nodes.

Slicer uses RAS for all scene data (meshes, fiducials...). BioFace3D's predict_landmarks
returns landmarks in the same coordinate system as the input mesh. When the mesh is
exported as LPS (as our module does), landmarks are LPS and must be converted to
RAS before adding to fiducials.
"""

import logging

import numpy as np

import slicer


# ----------------------------------------------------------------------------------------------------------------------
def create_fiducial_node():
    """Create a new fiducial node with default display settings. Returns vtkMRMLMarkupsFiducialNode."""
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
    node.SetName(slicer.mrmlScene.GenerateUniqueName("Landmarks"))
    node.GetDisplayNode().SetSelectedColor(0, 1, 0)
    node.SetLocked(False)
    return node


# ----------------------------------------------------------------------------------------------------------------------
def lps_to_ras(landmarks):
    """Convert landmarks from LPS to RAS (negate X and Y)."""
    lm = np.asarray(landmarks, dtype=np.float64)
    out = lm.copy()
    out[:, 0] = -lm[:, 0]
    out[:, 1] = -lm[:, 1]
    return out


# ----------------------------------------------------------------------------------------------------------------------
def landmarks_to_fiducial(landmarks, fiducial_node=None, labels=None, coordinate_system="RAS"):
    """
    Add landmark points to a fiducial node. Creates the node if None.

    Parameters
    ----------
    landmarks : numpy.ndarray
    fiducial_node : vtkMRMLMarkupsFiducialNode or None
    labels : list of str or None. Each label corresponds to a landmark.
    coordinate_system : str
    Returns: vtkMRMLMarkupsFiducialNode; fiducial node with control points added.
    """
    if landmarks is None or len(landmarks) == 0:
        if fiducial_node is None:
            return None
        fiducial_node.RemoveAllControlPoints()
        return fiducial_node

    landmarks = np.asarray(landmarks)
    if landmarks.ndim != 2 or landmarks.shape[1] != 3:
        raise ValueError("landmarks must be (N, 3) array, got shape {}".format(getattr(landmarks, "shape", "?")))

    if coordinate_system.upper() == "LPS":
        landmarks = lps_to_ras(landmarks)

    n = landmarks.shape[0]
    if labels is not None and len(labels) != n:
        raise ValueError("labels length {} != landmarks count {}".format(len(labels), n))

    if fiducial_node is None:
        fiducial_node = create_fiducial_node()
    else:
        fiducial_node.RemoveAllControlPoints()

    for i in range(n):
        label = (labels[i] if labels else "lm_{}".format(i + 1))
        fiducial_node.AddControlPoint(
            float(landmarks[i, 0]),
            float(landmarks[i, 1]),
            float(landmarks[i, 2]),
            label,
        )

    logging.info("landmarks_to_fiducial: added %d points to %s", n, fiducial_node.GetName())
    return fiducial_node
