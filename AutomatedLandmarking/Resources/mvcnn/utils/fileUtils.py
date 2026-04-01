import json
import numpy as np
import os
from collections import OrderedDict
from map import mapRaw as mr
from pathlib import Path
import re


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    with fname.open('rt') as f:
        return json.load(f, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as f:
        json.dump(content, f, indent=4, sort_keys=False)


def write_file(content, fname):
    with fname.open('wt') as f:
        f.write(content)


def append_to_file(content, fname):
    with fname.open('a') as f:
        f.write(content)


def subject_files_from_json(fname):

    # Read json file
    dataSrc = read_json(Path(fname))

    # Process every data block
    files = {}
    for dataBlock in dataSrc[mr.data]:

        # Get data block params
        dataPath = dataBlock[mr.data_path]
        data3dExt = dataBlock[mr.data_3d]
        dataLmExt = dataBlock[mr.data_lm]
        dataRgbExt = dataBlock[mr.data_rgb]
        dataList = dataBlock[mr.data_list]
        
        # Get precision file
        try:
            dataPrecExt = dataBlock[mr.data_prec]
        except:
            dataPrecExt = ''

        # Get subject files
        rawFiles = {}
        for data in dataList:
            files[data[mr.data_list_name]] = {
                'model': data3dExt,
                'lm': dataLmExt,
                'rgb': dataRgbExt if data[mr.data_list_ch][2] else '',
                'prec': dataPrecExt,
                'path': dataPath
            }

        # Check if any file exist
        if len(rawFiles) > 0:
            files.update(rawFiles)
    
    return files


def filter_subject_files(subjects_json_file, filter_ids):
    subjectFiles = subject_files_from_json(subjects_json_file)
    filteredFiles = {}
    for key in subjectFiles.keys():
        if key in filter_ids:
            filteredFiles[key] = subjectFiles[key]
    return filteredFiles


def mesh_files_in_dir(directory):
    names = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.obj', '.wrl', '.vtk', '.ply', '.stl')):
                full_name = os.path.join(root, filename)
                if os.path.isfile(full_name) and os.stat(full_name).st_size > 5:
                    names.append(full_name)
    return names


def read_landmarks(file_name):
    lms = []
    try:
        with open(file_name) as f:
            for line in f:
                line = line.strip('\n')
                x, y, z = np.double(line.split(' '))
                lms.append(np.array([x, y, z]))
        return np.array(lms)
    except:
        return None
    
def read_fcsv(file_path):
    # Read the content of the fcsv file
    with open(file_path, 'r') as file:
        content = file.read()

    # Use regular expressions to extract x, y, z columns
    matches = re.findall(r'(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)', content)

    # Convert list of tuples to numpy array
    landmarks = np.array([list(map(float, match)) for match in matches])

    return landmarks

def read_landmark_ascii(file_path):
    # Read the content of the landmarkAscii file
    with open(file_path, 'r') as file:
        content = file.read()

    # Use regular expressions to extract x, y, z columns
    matches = re.findall(r'(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)', content)

    # Convert list of tuples to numpy array
    landmarks = np.array([list(map(float, match)) for match in matches])

    return landmarks


def read_precision(file_name):
    try:
        with open(file_name) as f:
            for i, line in enumerate(f):
                if i == 1:
                    line = line.strip('\n')
                    spLine = line.split(',')
                    x, y, z = np.double(spLine[0:3])
                    units = spLine[3]
                    return x, y, z, units
                elif i > 1:
                    break
        return None, None, None, None
    except:
        return None, None, None, None


def read_lines(file_name):
    lines = []
    with open(file_name) as f:
        for line in f:
            line = line.strip('\n')
            lines.append(line)
    return lines
