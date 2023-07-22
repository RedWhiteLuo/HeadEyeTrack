import pathlib
import time
import cv2
import numpy as np
import os

import torch
from openvino.runtime import Core
from Skps.logger.logger import logger


class EyeCoordsInfer:
    def __init__(self, model_path_xml):
        self.model = torch.load(model_path_xml).eval()
        self.output_node = self.model.outputs[0]

    def __call__(self, image):
        image = image.transpose((2, 0, 1))
        image = image / 255  # [C, H, W] format
        output = self.model([image])[self.output_node]
        print(type(output))
        return 0


A = EyeCoordsInfer("./mymodule_1000.pth")
