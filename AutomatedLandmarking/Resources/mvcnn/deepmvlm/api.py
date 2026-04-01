from stat import FILE_ATTRIBUTE_NORMAL
import model.model as module_arch
import pathlib
import platform
import torch
from map import mapConfig as m
from prediction import Predict2D
from torch.utils.model_zoo import load_url
from utils import append_to_file
from utils3d import *


class DeepMVLM:
    def __init__(self, config):

        self.config = config
        self.logger = config.get_logger('predict')

        # Params for predictions on same file
        self.last_filename = None
        self.obj_data = None
        self.pd_data = None
        self.texture_data = None
        
        # Update path depending on platform
        if platform.system() == 'Windows':
            pathlib.PosixPath = pathlib.WindowsPath
        else:
            pathlib.WindowsPath = pathlib.PosixPath

        # Get model
        if self.config[m.pred][m.pred_mdl].find('http') == -1:
            self.device, self.model = self._get_device_and_load_model()
        else:
            self.device, self.model = self._get_device_and_load_model_from_url()


    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "prediction will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        if n_gpu_use > 0 and torch.cuda.is_available() \
                and (torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1] < 35):
            self.logger.warning("Warning: The GPU has lower CUDA capabilities than the required 3.5 - using CPU")
            n_gpu_use = 0
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids


    def _get_device_and_load_model_from_url(self):
        logger = self.config.get_logger('test')

        print('Initialising model')
        model = self.config.initialize(m.arch, module_arch)

        print('Loading checkpoint')
        model_dir = self.config[m.train][m.train_savedir] + "trained/"
        check_point_name = self.config[m.pred][m.pred_mdl]

        print('Getting device')
        device, device_ids = self._prepare_device(self.config[m.ngpu])

        logger.info('Loading checkpoint: {}'.format(check_point_name))
        checkpoint = load_url(check_point_name, model_dir, map_location=device)
        
        # Write clean model - should only be done once for translation of models
        # base_name = os.path.basename(os.path.splitext(check_point_name)[0])
        # clean_file = 'saved/trained/' + base_name + '_only_state_dict.pth'
        # torch.save(checkpoint['state_dict'], clean_file)

        state_dict = []
        # Hack until all dicts are transformed
        if check_point_name.find('only_state_dict') == -1:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return device, model


    # Deprecated - should not be used
    def _get_device_and_load_model(self):
        logger = self.config.get_logger('test')

        print('Initialising model')
        model = self.config.initialize(m.arch, module_arch)
        # logger.info(model)

        print('Loading checkpoint')
        check_point_name = self.config[m.pred][m.pred_mdl]

        print('Getting device')
        device, device_ids = self._prepare_device(self.config[m.ngpu])

        logger.info('Loading checkpoint: {}'.format(check_point_name))
        checkpoint = torch.load(check_point_name, map_location=device, weights_only=False)

        state_dict = []
        # Hack until all dicts are transformed
        if check_point_name.find('only_state_dict') == -1:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return device, model


    def _predict_file(self, file_name):

        # Clear previous data if required
        if self.last_filename is not None and self.last_filename != file_name:
            self.obj_data = None
            self.pd_data = None
            self.texture_data = None

        # Keep track of last file_name
        self.last_filename = file_name

        render_3d = Render3D(self.config)
        image_stack, transform_stack, self.obj_data, self.pd_data, self.texture_data = render_3d.render_3d_file(
            file_name, obj_data=self.obj_data, pd_data=self.pd_data, texture_data=self.texture_data)

        predict_2d = Predict2D(self.config, self.model, self.device)
        heatmap_maxima = predict_2d.predict_heatmaps_from_images(image_stack)

        u3d = Utils3D(self.config)
        u3d.heatmap_maxima = heatmap_maxima
        u3d.transformations_3d = transform_stack
        u3d.compute_lines_from_heatmap_maxima()
        #  u3d.visualise_one_landmark_lines(15)
        u3d.compute_all_landmarks_from_view_lines()
        u3d.project_landmarks_to_surface(file_name)

        return u3d.landmarks, u3d.ransac_error, u3d.lm_ransac_error


    def predict(self, file_name, basename, ko_file=None, output_path=None):
        # Prepare processing
        print('Processing ', file_name)
        if output_path is not None:
            name_lm_ascii = output_path / (basename + '.landmarkAscii')
            name_lm_txt = output_path / (basename + '.txt')
            name_lm_fcsv = output_path / (basename + '.fcsv')
            name_lm_json = output_path / (basename + '.json')
            name_re_txt = output_path / (basename + '_ransac.txt')
        else:
            name_lm_ascii = self.config.temp_dir / (basename + '_landmarks.landmarkAscii')
            name_lm_txt = self.config.temp_dir / (basename + '_landmarks.txt')
            name_lm_fcsv = self.config.temp_dir / (basename + '_landmarks.fcsv')
            name_lm_json = self.config.temp_dir / (basename + '_landmarks.json')
            name_re_txt = self.config.temp_dir / (basename + '_ransac.txt')

        # Prediction averaging
        landmarkList = []
        ransacList = []
        koCount = 0
        for i in range(self.config.predict_num):
            print('Prediction ', str(i + 1), ' out of ', self.config.predict_num)
            ransac_i = self.config.max_ransac
            while ransac_i >= self.config.max_ransac:
                landmarks_i, ransac_i, lm_ransac_i = self._predict_file(file_name)
                landmarkList.append(landmarks_i)
                ransacList.append(lm_ransac_i)
                if ransac_i >= self.config.max_ransac:
                    print('Invalid prediction on ', file_name)
                    koCount = koCount + 1
                    if koCount >= self.config.predict_tries:
                        if ko_file is not None:
                            append_to_file(file_name + '\n', ko_file)
                        print('Prediction will not be taken into account')
                        return None, None
                    else:
                        print('Retrying prediction')
        landmarks = row_mean(landmarkList)
        ransac = row_mean(ransacList)

        # Store results
        print('Storing ', file_name, ' results on ', str(self.config.temp_dir))
        if self.config.output_format == 'txt':
            write_landmarks_as_txt(landmarks, name_lm_txt)
        elif self.config.output_format == 'landmarkAscii':
            write_landmarks_as_ascii(landmarks, name_lm_ascii)
        elif self.config.output_format == 'fcsv':
            write_landmarks_as_fcsv(landmarks, name_lm_fcsv)
        elif self.config.output_format == 'json':
            write_landmarks_as_json(landmarks, name_lm_json)
        elif self.config.output_format == 'all':
            write_landmarks_as_txt(landmarks, name_lm_txt)
            write_landmarks_as_ascii(landmarks, name_lm_ascii)
            write_landmarks_as_fcsv(landmarks, name_lm_fcsv)
            write_landmarks_as_json(landmarks, name_lm_json)
        else:
            write_landmarks_as_txt(landmarks, name_lm_txt)
        # write_ransac_as_txt(ransac, name_re_txt)
        return landmarks, ransac
