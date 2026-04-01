from utils3d import *
import open3d as o3d
import numpy as np
import glob, os
import sys
import vtk

if __name__ == "__main__":
    
    ply_file = sys.argv[1]
    out_path = sys.argv[2]
    #Generate projection
    ren = vtk.vtkRenderer()
    ren.SetBackground(0, 0, 0)
    visualise_mesh_and_landmarks(ply_file, off_screen=True, out_file=out_path, renderer=ren)
