from utils3d import *
from utils import read_landmarks, read_fcsv, read_landmark_ascii
import sys

if __name__ == "__main__":


    if len(sys.argv) >= 3:

        ply_file = sys.argv[1]
        land_file = sys.argv[2]
        print(sys.argv[1])
        # Get the file extension
        file_extension = land_file.split('.')[-1].lower()

        # Call the corresponding function based on the file extension
        if file_extension == 'fcsv':
            landmarks = read_fcsv(land_file)
        elif file_extension == 'landmarkascii':
            landmarks =  read_landmark_ascii(land_file)
        elif file_extension == 'txt':
            landmarks = read_landmarks(land_file)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        # landmarks = read_landmarks(txt_file)
        visualise_mesh_and_landmarks(ply_file, landmarks)
    else:
        ply_file = sys.argv[1]
        visualise_mesh_and_landmarks(ply_file)