import copy
import math
import numpy as np
import time
import torch
from map import mapConfig as m


class Predict2D:
    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device


    def find_heat_map_maxima(self, heatmaps, sigma=None, method="simple"):
        """ heatmaps: (#LM, hm_size,hm_size) """
        out_dim = heatmaps.shape[0]  # number of landmarks
        hm_size = heatmaps.shape[1]
        # coordinates = np.zeros((out_dim, 2), dtype=np.float32)
        coordinates = np.zeros((out_dim, 3), dtype=np.float32)

        # TODO Need to figure out why x and y are switched here...probably something with row, col
        # simple: Use only maximum pixel value in HM
        if method == "simple":
            for k in range(out_dim):
                hm = copy.copy(heatmaps[k, :, :])
                highest_idx = np.unravel_index(np.argmax(hm), (hm_size, hm_size))
                px = highest_idx[0]
                py = highest_idx[1]
                value = hm[px, py]  # TODO check if values is equal to np.max(hm)
                coordinates[k, :] = (px - 1, py - 0.5, value)  # TODO find out why it works with the subtractions

        if method == "moment":
            for k in range(out_dim):
                hm = heatmaps[k, :, :]
                highest_idx = np.unravel_index(np.argmax(hm), (hm_size, hm_size))
                px = highest_idx[0]
                py = highest_idx[1]

                value = np.max(hm)

                # Size of window around max (15 on each side gives an array of 2 * 5 + 1 values)
                sz = 15
                a_len = 2 * sz + 1
                if px > sz and hm_size-px > sz and py > sz and hm_size-py > sz:
                    slc = hm[px-sz:px+sz+1, py-sz:py+sz+1]
                    ar = np.arange(a_len)
                    sum_x = np.sum(slc, axis=1)
                    s = np.sum(np.multiply(ar, sum_x))
                    ss = np.sum(sum_x)
                    pos = s / ss - sz
                    px = px + pos

                    sum_y = np.sum(slc, axis=0)
                    s = np.sum(np.multiply(ar, sum_y))
                    ss = np.sum(sum_y)
                    pos = s / ss - sz
                    py = py + pos

                coordinates[k, :] = (px-1, py-0.5, value)  # TODO find out why it works with the subtractions

        return coordinates


    def find_maxima_in_batch_of_heatmaps(self, heatmaps, cur_id, heatmap_maxima):
        write_heatmaps = False
        heatmaps = heatmaps.numpy()
        batch_size = heatmaps.shape[0]

        f = None
        for idx in range(batch_size):
            if write_heatmaps:
                name_hm_maxima = self.config.temp_dir / ('hm_maxima' + str(cur_id + idx) + '.txt')
                f = open(name_hm_maxima, 'w')

            coordinates = self.find_heat_map_maxima(heatmaps[idx, :, :, :], method='moment')
            for lm_no in range(coordinates.shape[0]):
                px = coordinates[lm_no][0]
                py = coordinates[lm_no][1]
                value = coordinates[lm_no][2]
                if value > 1.2:  # TODO debug - really bad hack due to weird max in heatmaps
                    print("Found heatmap with value > 1.2 LM {} value {} pos {} {}  ".format(lm_no, value, px, py))
                    value = 0
                heatmap_maxima[lm_no, cur_id + idx, :] = (px, py, value)
                if write_heatmaps:
                    out_str = str(px) + ' ' + str(py) + ' ' + str(value) + '\n'
                    f.write(out_str)

            if write_heatmaps:
                f.close()

    def predict_heatmaps_from_images(self, image_stack):
        n_views = self.config[m.dl][m.dl_args][m.dl_args_views]
        batch_size = self.config[m.dl][m.dl_args][m.dl_args_batchsize]
        n_landmarks = self.config[m.arch][m.arch_args][m.arch_args_nlm]

        write_heatmaps = False  # Debug only - removed for minimal/Slicer use
        heatmap_maxima = np.zeros((n_landmarks, n_views, 3))

        print('Predicting heatmaps for all views')
        start = time.time()
        cur_id = 0
        while cur_id + batch_size <= n_views:
            cur_images = image_stack[cur_id:cur_id + batch_size, :, :, :]

            data = torch.from_numpy(cur_images)
            data = data.permute(0, 3, 1, 2)  # from NHWC to NCHW

            with torch.no_grad():
                data = data.to(self.device)
                output = self.model(data)

                # output [stack (0 or 1), batch, lm, hm_size, hm_size]
                heatmaps = output[1, :, :, :, :].cpu()
                self.find_maxima_in_batch_of_heatmaps(heatmaps, cur_id, heatmap_maxima)

            cur_id = cur_id + batch_size

        end = time.time()
        print("Model prediction time: " + str(end - start))
        return heatmap_maxima
