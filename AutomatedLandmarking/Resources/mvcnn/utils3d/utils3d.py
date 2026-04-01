import numpy as np
import os
import vtk
from map import mapConfig as m
from utils import write_file


def apply_pre_transformation(pal_config, tmp_dir, pd):
    translation = [0, 0, 0]
    if pal_config[m.pal_com]:
        vtk_cm = vtk.vtkCenterOfMass()
        vtk_cm.SetInputData(pd)
        vtk_cm.SetUseScalarsAsWeights(False)
        vtk_cm.Update()
        cm = vtk_cm.GetCenter()
        translation = [-cm[0], -cm[1], -cm[2]]

    t = vtk.vtkTransform()
    t.Identity()

    rx = pal_config[m.pal_rx]
    ry = pal_config[m.pal_ry]
    rz = pal_config[m.pal_rz]
    s = pal_config[m.pal_scale]

    t.Scale(s, s, s)
    t.RotateY(ry)
    t.RotateX(rx)
    t.RotateZ(rz)
    t.Translate(translation)
    t.Update()

    # Transform (assuming only one mesh)
    trans = vtk.vtkTransformPolyDataFilter()
    trans.SetInputData(pd)
    trans.SetTransform(t)
    trans.Update()

    if pal_config[m.pal_write]:
        name_out = str(tmp_dir / ('pre_transform_mesh.vtk'))
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(trans.GetOutput())
        writer.SetFileName(name_out)
        writer.Write()

    return trans.GetOutput(), t


def read_surface(file_name):
    pd_in = None
    file_extension = os.path.splitext(file_name)[1]
    if file_extension == ".wrl":
        pd_in = vtk.vtkVRMLImporter()
        pd_in.SetFileName(file_name)
        pd_in.Update()
        return pd_in.GetRenderer().GetActors().GetLastActor().GetMapper().GetInput()
    elif file_extension == ".obj":
        pd_in = vtk.vtkOBJReader()
    elif file_extension == ".vtk":
        pd_in = vtk.vtkPolyDataReader()
    elif file_extension == ".stl":
        pd_in = vtk.vtkSTLReader()
    elif file_extension == ".ply":
        pd_in = vtk.vtkPLYReader()
    else:
        print("Can not read files with extension", file_extension)
        return None
    pd_in.SetFileName(file_name)
    pd_in.Update()
    return pd_in.GetOutput()


def read_texture_img(file_name):

    # Get valid image file
    real_file_name = None
    img_texture = os.path.splitext(file_name)[0] + ".bmp"
    if os.path.isfile(img_texture):
        real_file_name = img_texture
    img_texture = os.path.splitext(file_name)[0] + ".png"
    if os.path.isfile(img_texture):
        real_file_name = img_texture
    img_texture = os.path.splitext(file_name)[0] + ".jpg"
    if os.path.isfile(img_texture):
        real_file_name = img_texture
        
    # Load texture
    if real_file_name is not None:
        texture_img = None
        file_extension = os.path.splitext(real_file_name)[1]
        if file_extension == ".bmp":
            texture_img = vtk.vtkBMPReader()
        elif file_extension == ".png":
            texture_img = vtk.vtkPNGReader()
        elif file_extension == ".jpg":
            texture_img = vtk.vtkJPEGReader()
        else:
            return None
        texture_img.SetFileName(real_file_name)
        texture_img.Update()
        return texture_img

    return None


def read_texture_output(file_name):
    texture_img = read_texture_img(file_name)
    if texture_img is not None:
        return texture_img.GetOutput()
    return None


def write_landmarks_as_ascii(landmarks, file_name):
    # Set header
    content = '# AmiraMesh 3D ASCII 2.0\n\n\n'

    # Set markers count
    content = content + 'define Markers {:d}\n\n'.format(len(landmarks))

    # Set parameters
    content = content + 'Parameters {\n    NumSets 1,\n    ContentType "LandmarkSet"\n}\n\n'

    # Set marker format
    content = content + 'Markers { float[3] Coordinates } @1\n\n'

    # Set data section
    content = content + '# Data section follows\n@1\n'
    for lm in landmarks:
        px = lm[0]
        py = lm[1]
        pz = lm[2]
        content = content + '{:.3f} {:.3f} {:.3f}\n'.format(px, py, pz)
    write_file(content, file_name)


def write_landmarks_as_txt(landmarks, file_name):
    content = ''
    for lm in landmarks:
        px = lm[0]
        py = lm[1]
        pz = lm[2]
        content = content + '{:.3f} {:.3f} {:.3f}\n'.format(px, py, pz)
    write_file(content, file_name)


def write_landmarks_as_fcsv(landmarks, file_name):
    # Fixed header of the FCSV file
    fcsv_header = [
        '# Markups fiducial file version = 4.11\n',
        '# CoordinateSystem = LPS\n',
        '# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n'
    ]

    # Create a list to store the lines of the FCSV file
    fcsv_lines = []

    # Add header lines to the list
    fcsv_lines.extend(fcsv_header)

    # Add landmark lines to the list
    x_columns_id = 0
    for lm in landmarks:
        # Modify the x_Columns_Id
        x_columns_id = x_columns_id + 1
        # Modify the XYZ coordinates of the landmark
        x = lm[0]
        y = lm[1]
        z = lm[2]
        # Modify coordinates
        ow, ox, oy, oz = 0, 0, 0, 1
        vis, sel, lock = 1, 1, 0
        # Modify the label of the landmark
        label = f'F-{x_columns_id}'
        # Remove NaN's from the "desc" and "associatedNodeID" columns
        desc, associated_node_id = [], []

        # Create a line for the current landmark
        line = f"{x_columns_id},{x},{y},{z},{ow},{ox},{oy},{oz},{vis},{sel},{lock},{label},{desc},{associated_node_id}\n"

        # Add the line to the list
        fcsv_lines.append(line)

    # Write the FCSV lines to the output file
    write_file(''.join(fcsv_lines), file_name)


def write_landmarks_as_json(landmarks, file_name):
    """Save landmarks as JSON for Slicer integration.
    Format: {"landmarks": [[x,y,z], ...], "count": N}
    """
    import json
    data = {
        'landmarks': landmarks.tolist() if hasattr(landmarks, 'tolist') else [list(lm) for lm in landmarks],
        'count': len(landmarks)
    }
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=2)


def write_ransac_as_txt(lm_ransac, file_name):
    content = ''
    for lm_r in lm_ransac:
        content = content + '{:.3f}\n'.format(lm_r[0])
    write_file(content, file_name)


def row_mean(matrix):
    sum = 0
    for i in range(len(matrix)):
        curRow = np.array(matrix[i])
        sum = sum + curRow
    avg = sum / len(matrix)
    return avg


def landmark_error(predict_lm, expected_lm):
    error = np.empty([len(predict_lm), 1])
    for i in range(len(predict_lm)):
        error[i] = np.linalg.norm(expected_lm[i] - predict_lm[i])
    return error


class Utils3D:
    def __init__(self, config):
        self.config = config
        self.heatmap_maxima = None
        self.transformations_3d = None
        self.lm_start = None
        self.lm_end = None
        self.landmarks = None
        self.logger = config.get_logger('Utils3D')


    def read_heatmap_maxima(self, dir_name=None):
        if dir_name is None:
            dir_name = str(self.config.temp_dir)
        print('Reading from', dir_name)

        n_landmarks = self.config[m.arch][m.arch_args][m.arch_args_nlm]
        n_views = self.config[m.dl][m.dl_args][m.dl_args_views]

        # [n_landmarks, n_views, x, y, value]
        self.heatmap_maxima = np.zeros((n_landmarks, n_views, 3))

        for idx in range(n_views):
            name_hm_maxima = dir_name + '/hm_maxima' + str(idx) + '.txt'
            with open(name_hm_maxima) as f:
                id_lm = 0
                for line in f:
                    line = line.strip("/n")
                    x, y, val = np.double(line.split(" "))
                    self.heatmap_maxima[id_lm, idx, :] = (x, y, val)
                    id_lm = id_lm + 1
                    if id_lm > n_landmarks:
                        print('Too many landmarks in file ', name_hm_maxima)
                        break

            if id_lm != n_landmarks:
                print('Too few landmarks in file ', name_hm_maxima)


    def read_3d_transformations(self, dir_name=None):
        if dir_name is None:
            dir_name = str(self.config.temp_dir)
        print('Reading from', dir_name)

        n_views = self.config[m.dl][m.dl_args][m.dl_args_views]

        # [n_views, rx, ry, rz, s, tx, ty]
        self.transformations_3d = np.zeros((n_views, 6))

        for idx in range(n_views):
            name_hm_maxima = dir_name + '/transform' + str(idx) + '.txt'
            rx, ry, rz, s, tx, ty = np.loadtxt(name_hm_maxima)
            self.transformations_3d[idx, :] = (rx, ry, rz, s, tx, ty)


    # Each maxima in a heatmap corresponds to a line in 3D space of the original 3D shape
    # This function transforms the maxima to (start point, end point) pairs
    def compute_lines_from_heatmap_maxima(self):
        n_landmarks = self.heatmap_maxima.shape[0]
        n_views = self.heatmap_maxima.shape[1]

        self.lm_start = np.zeros((n_landmarks, n_views, 3))
        self.lm_end = np.zeros((n_landmarks, n_views, 3))

        img_size = self.config[m.dl][m.dl_args][m.dl_args_imgsize]
        hm_size = self.config[m.dl][m.dl_args][m.dl_args_hmapsize]
        winsize = img_size

        # TODO these fixed values should probably be in a config file
        x_min = -150
        x_max = 150
        y_min = -150
        y_max = 150
        x_len = x_max - x_min
        y_len = y_max - y_min

        pd = vtk.vtkPolyData()
        for idx in range(n_views):
            rx, ry, rz, s, tx, ty = self.transformations_3d[idx, :]

            # Set transformation matrix in vtk
            t = vtk.vtkTransform()
            t.Identity()
            t.Update()

            t.Identity()
            t.RotateY(ry)
            t.RotateX(rx)
            t.RotateZ(rz)
            t.Update()

            for lm_no in range(n_landmarks):
                # [n_landmarks, n_views, x, y, value]
                y = self.heatmap_maxima[lm_no, idx, 0]
                x = self.heatmap_maxima[lm_no, idx, 1]
                # value = self.heatmap_maxima[lm_no, idx, 2]

                #  Extract just one landmark and scale it according to heatmap and image sizes
                y = y / hm_size * img_size
                x = x / hm_size * img_size

                # Making end points of line in world coordinates
                p_wc_s = np.zeros((3, 1))
                p_wc_e = np.zeros((3, 1))

                p_wc_s[0] = (x / winsize) * x_len + x_min
                p_wc_s[1] = ((winsize - 1 - y) / winsize) * y_len + y_min
                p_wc_s[2] = 500

                p_wc_e[0] = (x / winsize) * x_len + x_min
                p_wc_e[1] = ((winsize - 1 - y) / winsize) * y_len + y_min
                p_wc_e[2] = -500

                # Insert line into vtk-framework to transform
                points = vtk.vtkPoints()
                lines = vtk.vtkCellArray()

                lines.InsertNextCell(2)
                pid = points.InsertNextPoint(p_wc_s)
                lines.InsertCellPoint(pid)
                pid = points.InsertNextPoint(p_wc_e)
                lines.InsertCellPoint(pid)

                pd.SetPoints(points)
                del points
                pd.SetLines(lines)
                del lines

                # Do inverse transform into original space
                tfilt = vtk.vtkTransformPolyDataFilter()
                tfilt.SetTransform(t.GetInverse())
                tfilt.SetInputData(pd)
                tfilt.Update()

                lm_out = vtk.vtkPolyData()
                lm_out.DeepCopy(tfilt.GetOutput())

                self.lm_start[lm_no, idx, :] = lm_out.GetPoint(0)
                self.lm_end[lm_no, idx, :] = lm_out.GetPoint(1)

                del tfilt
            del t
        del pd


    def visualise_one_landmark_lines(self, lm_no, dir_name=None):
        if dir_name is None:
            dir_name = str(self.config.temp_dir)
        print('Writing to', dir_name)

        lm_name = dir_name + '/lm_lines_' + str(lm_no) + '.vtk'

        n_views = self.heatmap_maxima.shape[1]
        pd = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        verts = vtk.vtkCellArray()
        scalars = vtk.vtkDoubleArray()
        scalars.SetNumberOfComponents(1)
        scalars.SetNumberOfValues(2 * n_views)

        for idx in range(n_views):
            lines.InsertNextCell(2)
            pid = points.InsertNextPoint(self.lm_start[lm_no, idx, :])
            lines.InsertCellPoint(pid)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(pid)
            pid = points.InsertNextPoint(self.lm_end[lm_no, idx, :])
            lines.InsertCellPoint(pid)
            scalars.SetValue(idx * 2, self.heatmap_maxima[lm_no, idx, 2])  # Color code according to maxima value
            scalars.SetValue(idx * 2 + 1, self.heatmap_maxima[lm_no, idx, 2])

        pd.SetPoints(points)
        del points
        pd.SetLines(lines)
        del lines
        pd.SetVerts(verts)
        del verts
        pd.GetPointData().SetScalars(scalars)
        del scalars

        writer = vtk.vtkPolyDataWriter()
        writer.SetInputData(pd)
        writer.SetFileName(lm_name)
        writer.Write()

        del writer
        del pd


    # FROM: https://se.mathworks.com/matlabcentral/fileexchange/37192-intersection-point-of-lines-in-3d-space?focused
    # =5235003&tab=function"
    """
    Find intersection point of lines in 3D space, in the least squares sense.
    pa :          Nx3-matrix containing starting point of N lines
    pa :          Nx3-matrix containing end point of N lines
    p_intersect : Best intersection point of the N lines, in least squares sense.
    distances   : Distances from intersection point to the input lines
    Anders Eikenes, 2012 """
    def compute_intersection_between_lines(self, pa, pb):
        n_lines = pa.shape[0]
        si = pb - pa  # N lines described as vectors
        ni = np.divide(si, np.transpose(np.sqrt(np.sum(si ** 2, 1)) * np.ones((3, n_lines))))  # Normalize vectors
        nx = ni[:, 0]
        ny = ni[:, 1]
        nz = ni[:, 2]
        sxx = np.sum(nx ** 2 - 1)
        syy = np.sum(ny ** 2 - 1)
        szz = np.sum(nz ** 2 - 1)
        sxy = np.sum(np.multiply(nx, ny))
        sxz = np.sum(np.multiply(nx, nz))
        syz = np.sum(np.multiply(ny, nz))
        s = np.array([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
        cx = np.sum(np.multiply(pa[:, 0], (nx ** 2 - 1)) + np.multiply(pa[:, 1], np.multiply(nx, ny)) +
                    np.multiply(pa[:, 2], np.multiply(nx, nz)))
        cy = np.sum(np.multiply(pa[:, 0], np.multiply(nx, ny)) + np.multiply(pa[:, 1], (ny ** 2 - 1)) +
                    np.multiply(pa[:, 2], np.multiply(ny, nz)))
        cz = np.sum(np.multiply(pa[:, 0], np.multiply(nx, nz)) + np.multiply(pa[:, 1], np.multiply(ny, nz)) +
                    np.multiply(pa[:, 2], (nz ** 2 - 1)))

        c = np.array([[cx], [cy], [cz]])
        p_intersect = np.matmul(np.linalg.pinv(s), c)
        return p_intersect[:, 0]


    def compute_intersection_between_lines_ransac(self, pa, pb):
        # TODO parameters in config
        iterations = 100
        best_error = 100000000  # TODO should find a better initialiser
        best_p = (0, 0, 0)
        dist_thres = 10 * 10  # TODO should find a better way to esimtate dist_thres
        # d = 10  #
        n_lines = len(pa)
        d = n_lines / 3
        used_lines = -1

        for i in range(iterations):
            # get 3 random lines
            ran_lines = np.random.choice(range(n_lines), 3, replace=False)
            # Compute first estimate of intersection
            p_est = self.compute_intersection_between_lines(pa[ran_lines, :], pb[ran_lines, :])
            # Compute distance from all lines to intersection
            top = np.cross((np.transpose(p_est) - pa), (np.transpose(p_est) - pb))
            bottom = pb - pa
            distances = (np.linalg.norm(top, axis=1) / np.linalg.norm(bottom, axis=1))**2
            # number of inliners
            n_inliners = np.sum(distances < dist_thres)
            if n_inliners > d:
                # reestimate based on inliners
                idx = distances < dist_thres
                p_est = self.compute_intersection_between_lines(pa[idx, :], pb[idx, :])

                # Compute distance from all inliners to intersection
                top = np.cross((np.transpose(p_est) - pa[idx, :]), (np.transpose(p_est) - pb[idx, :]))
                bottom = pb[idx, :] - pa[idx, :]
                distances = (np.linalg.norm(top, axis=1) / np.linalg.norm(bottom, axis=1))**2

                # sum_squared = np.sum(np.square(distances)) / n_inliners
                sum_squared = np.sum(distances) / n_inliners
                if sum_squared < best_error:
                    best_error = sum_squared
                    best_p = p_est
                    used_lines = n_inliners

        if used_lines == -1:
            self.logger.warning('Ransac failed - estimating from all lines')
            best_p = self.compute_intersection_between_lines(pa, pb)
        # else:
        # print('Ransac error ', best_error, ' with ', used_lines, ' lines')

        return best_p, best_error


    # return the lines that correspond to a high valued maxima in the heatmap
    def filter_lines_based_on_heatmap_value_using_quantiles(self, lm_no, pa, pb):
        max_values = self.heatmap_maxima[lm_no, :, 2]
        q = self.config[m.mdl][m.mdl_maxq]
        threshold = np.quantile(max_values, q)
        idx = max_values > threshold
        # print('Using ', threshold, ' as threshold in heatmap maxima')
        pa_new = pa[idx]
        pb_new = pb[idx]
        return pa_new, pb_new


    # return the lines that correspond to a high valued maxima in the heatmap
    def filter_lines_based_on_heatmap_value_using_absolute_value(self, lm_no, pa, pb):
        max_values = self.heatmap_maxima[lm_no, :, 2]
        threshold = self.config[m.mdl][m.mdl_thresh]
        idx = max_values > threshold
        pa_new = pa[idx]
        pb_new = pb[idx]
        return pa_new, pb_new


    # Each landmark can be computed by the intersection of the view lines going trough (or near) it
    def compute_all_landmarks_from_view_lines(self):
        n_landmarks = self.heatmap_maxima.shape[0]
        self.landmarks = np.zeros((n_landmarks, 3))

        sum_error = 0
        self.lm_ransac_error = np.empty([n_landmarks, 1])
        for lm_no in range(n_landmarks):
            pa = self.lm_start[lm_no, :, :]
            pb = self.lm_end[lm_no, :, :]
            if self.config[m.mdl][m.mdl_vlines] == "abs_value":
                pa, pb = self.filter_lines_based_on_heatmap_value_using_absolute_value(lm_no, pa, pb)
            elif self.config[m.mdl][m.mdl_vlines] == "quantile":
                pa, pb = self.filter_lines_based_on_heatmap_value_using_quantiles(lm_no, pa, pb)
            p_intersect = (0, 0, 0)
            if len(pa) < 3:
                print('Not enough valid view lines for landmark ', lm_no)
            else:
                # p_intersect = self.compute_intersection_between_lines(pa, pb)
                p_intersect, best_error = self.compute_intersection_between_lines_ransac(pa, pb)
                sum_error = sum_error + best_error
                self.lm_ransac_error[lm_no] = best_error
            self.landmarks[lm_no, :] = p_intersect
        print("Ransac average error ", sum_error/n_landmarks)
        self.ransac_error = sum_error / n_landmarks


    def transform_landmarks_to_original_space(self, landmarks, t):
        points = vtk.vtkPoints()
        pd = vtk.vtkPolyData()
        # verts = vtk.vtkCellArray()

        for lm in landmarks:
            pid = points.InsertNextPoint(lm)
            # verts.InsertNextCell(1)
            # verts.InsertCellPoint(pid)
        pd.SetPoints(points)

        trans = vtk.vtkTransformPolyDataFilter()
        trans.SetInputData(pd)
        trans.SetTransform(t.GetInverse())
        trans.Update()
        pd_trans = trans.GetOutput()

        n_landmarks = pd_trans.GetNumberOfPoints()
        new_landmarks = np.zeros((n_landmarks, 3))
        for lm_no in range(pd_trans.GetNumberOfPoints()):
            p = pd_trans.GetPoint(lm_no)
            new_landmarks[lm_no, :] = (p[0], p[1], p[2])
        return new_landmarks


    # Project found landmarks to closest point on the target surface
    # return the landmarks in the original space
    def project_landmarks_to_surface(self, mesh_name):
        pd = read_surface(mesh_name)

        pd, t = apply_pre_transformation(self.config[m.pal], self.config.temp_dir, pd)

        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(pd)
        # clean.SetInputConnection(pd.GetOutputPort())
        clean.Update()

        locator = vtk.vtkCellLocator()
        locator.SetDataSet(clean.GetOutput())
        locator.SetNumberOfCellsPerBucket(1)
        locator.BuildLocator()

        projected_landmarks = np.copy(self.landmarks)
        n_landmarks = self.landmarks.shape[0]

        for i in range(n_landmarks):
            p = self.landmarks[i, :]
            cell_id = vtk.mutable(0)
            sub_id = vtk.mutable(0)
            dist2 = vtk.reference(0)
            tcp = np.zeros(3)

            locator.FindClosestPoint(p, tcp, cell_id, sub_id, dist2)
            # print('Nearest point in distance ', np.sqrt(np.float(dist2)))
            projected_landmarks[i, :] = tcp

        # self.landmarks = projected_landmarks
        self.landmarks = self.transform_landmarks_to_original_space(projected_landmarks, t)

        del pd
        del clean
        del locator
