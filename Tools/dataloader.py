import os
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


def create_data_source(path, with_annot=False):
    """
    :param path: img root path
    :param with_annot: whether the return should contain annotation
    :return: if with annot : return [N, (1CHW, annot)] else return [N, 1, 3, 32, 128]
    """
    data = []
    all_file_name = os.listdir(path)
    if len(all_file_name) < 1:
        raise ValueError('empty path')
    for file_name in all_file_name:
        file_path = path + file_name
        img = cv2.imread(file_path)
        img = img.transpose(2, 0, 1)
        img = img[None]
        if with_annot:
            coords = [int(file_name.split('_')[1]) / 1920, int(file_name.split('_')[2][:-4]) / 1080]
            data.append((torch.tensor(img), coords))
        else:
            data.append(torch.tensor(img))
    return data
