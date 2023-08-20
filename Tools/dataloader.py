import os
import random
import cv2
import numpy as np
import torch


def EpochDataLoader(path, batch_size=64):
    epoch_img, epoch_coords = [], []
    all_file_name = os.listdir(path)  # get all file name -> list
    file_num = len(all_file_name)
    batch_num = file_num // batch_size
    for i in range(batch_num):  # how many batch
        curr_batch = all_file_name[batch_size * i:batch_size * (i + 1)]
        batch_img, batch_coords = [], []
        for file_name in curr_batch:
            img = cv2.imread(str(path) + "/" + str(file_name))  # [H, W, C] format
            img = img.transpose((2, 0, 1))
            img = img / 255  # [C, H, W] format
            coords = [int(file_name.split('_')[1]) / 1920, int(file_name.split('_')[2][:-4]) / 1080]  # [x, y]
            batch_img.append(img)
            batch_coords.append(coords)
        epoch_img.append(batch_img)
        epoch_coords.append(batch_coords)
    epoch_img = torch.from_numpy(np.array(epoch_img)).float()
    epoch_coords = torch.from_numpy(np.array(epoch_coords)).float()
    return epoch_img, epoch_coords
