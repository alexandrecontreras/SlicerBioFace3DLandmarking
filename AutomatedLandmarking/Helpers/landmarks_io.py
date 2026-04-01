"""
Save landmark arrays to file formats (txt, fcsv, landmarkAscii).
Landmarks are (N, 3) numpy arrays, typically in LPS.
"""

import numpy as np


def write_landmarks_txt(landmarks, path):
    """Plain text: x y z per line."""
    lm = np.asarray(landmarks)
    path = str(path)
    with open(path, "w") as f:
        for i in range(len(lm)):
            f.write("{:.3f} {:.3f} {:.3f}\n".format(lm[i, 0], lm[i, 1], lm[i, 2]))


def write_landmarks_fcsv(landmarks, path):
    """Slicer fiducial format (LPS)."""
    lm = np.asarray(landmarks)
    path = str(path)
    lines = [
        "# Markups fiducial file version = 4.11\n",
        "# CoordinateSystem = LPS\n",
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n",
    ]
    for i in range(len(lm)):
        x, y, z = lm[i, 0], lm[i, 1], lm[i, 2]
        label = "F-%d" % (i + 1)
        lines.append("%d,%.3f,%.3f,%.3f,0,0,0,1,1,1,0,%s,[],[]\n" % (i + 1, x, y, z, label))
    with open(path, "w") as f:
        f.writelines(lines)


def write_landmarks_ascii(landmarks, path):
    """AmiraMesh landmarkAscii format."""
    lm = np.asarray(landmarks)
    path = str(path)
    lines = [
        "# AmiraMesh 3D ASCII 2.0\n\n\n",
        "define Markers %d\n\n" % len(lm),
        "Parameters {\n    NumSets 1,\n    ContentType \"LandmarkSet\"\n}\n\n",
        "Markers { float[3] Coordinates } @1\n\n",
        "# Data section follows\n@1\n",
    ]
    for i in range(len(lm)):
        lines.append("{:.3f} {:.3f} {:.3f}\n".format(lm[i, 0], lm[i, 1], lm[i, 2]))
    with open(path, "w") as f:
        f.writelines(lines)
