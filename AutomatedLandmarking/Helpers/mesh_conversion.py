"""
Mesh export for BioFace3D prediction.

Exports Slicer vtkMRMLModelNode to PLY for predict_landmarks (single mode).
"""

import logging
import os

import vtk

import slicer


def slicer_mesh_to_ply(mesh_node, output_path):
    """
    Export a Slicer vtkMRMLModelNode to a PLY file.
    """
    if mesh_node is None:
        raise ValueError("mesh_node is None")
    output_path = os.path.abspath(os.path.expanduser(str(output_path)))
    poly_data = mesh_node.GetPolyData()
    if poly_data is None or poly_data.GetNumberOfPoints() == 0:
        raise ValueError("Mesh node has no poly data or no points")

    # Undo Slicer's LPS→RAS conversion
    transform = vtk.vtkTransform()
    transform.Scale(-1.0, -1.0, 1.0)
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(poly_data)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    poly_to_write = transform_filter.GetOutput()

    writer = vtk.vtkPLYWriter()
    writer.SetInputData(poly_to_write)
    writer.SetFileName(output_path)
    writer.SetFileTypeToBinary()
    writer.Write()
    if not os.path.isfile(output_path):
        raise IOError("PLY file was not written: {}".format(output_path))
    logging.info("slicer_mesh_to_ply: wrote %s", output_path)
    return output_path
