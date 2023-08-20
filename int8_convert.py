import os
import cv2
import nncf
import torch
import openvino
import numpy as np
from sklearn.metrics import mean_squared_error
from openvino.runtime import Core, serialize  # serialize actually exist

DATASET_PATH = "E:/AI_Dataset/0Project/HeadEyeTrack/"
INPUT_MODEL_PATH = "Model/ET-last-FP32.xml"
OUTPUT_MODEL_PATH = "Model/ET-last-INT8.xml"


def create_data_source(path=None, with_annot=False):
    """
    :param path: img root path
    :param with_annot: whether the return should contain annotation
    :return: if with annot : return [N, (1CHW, annot)] else return [N, 1, 3, 32, 128]
    """
    data = []
    path = path if path else DATASET_PATH
    all_file_name = os.listdir(path)
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
    return data  # [N, 1, 3, 32, 128]


def transform_fn(data_item):
    # convert input tensor into float format
    images = data_item.float()
    # scale input
    images = images / 255
    # convert torch tensor to numpy array
    images = images.cpu().detach().numpy()
    return images


def transform_fn_with_annot(data_item):
    images = data_item.float()[0]
    images = images / 255
    images = images.cpu().detach().numpy()
    return images


def validate(model, validation_loader) -> float:
    """
    :param model: Model that need to be quantized
    :param validation_loader: the func 'create_data_source(path=None, with_annot=True)', return (image, target) in 'for'
    :return: float MSE between annot and predict
    """
    predictions, references = [], []
    output = model.outputs[0]
    for images, target in validation_loader:
        pred = model(images)[output][0]
        predictions.append(pred)
        references.append(target)
    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return mean_squared_error(predictions, references)


def basic_quantization():
    # prepare required data
    data_source = create_data_source()
    nncf_calibration_dataset = nncf.Dataset(data_source, transform_fn)
    # set the parameter of how to quantize
    subset_size = 1000
    preset = nncf.QuantizationPreset.PERFORMANCE  # or you can choose: nncf.QuantizationPreset.MIXED
    # load Model
    ov_model = Core().read_model(INPUT_MODEL_PATH)
    # perform quantize
    quantized_model = nncf.quantize(ov_model, nncf_calibration_dataset, preset=preset, subset_size=subset_size)
    # save Model
    serialize(quantized_model, OUTPUT_MODEL_PATH)


def accuracy_quantization():
    # prepare required data
    calibration_source = create_data_source(with_annot=False)
    validation_source = create_data_source(with_annot=True)
    calibration_dataset = nncf.Dataset(calibration_source, transform_fn)
    validation_dataset = nncf.Dataset(validation_source, transform_fn_with_annot)
    # load Model
    ov_model = Core().read_model(INPUT_MODEL_PATH)
    # perform quantize
    quantized_model = nncf.quantize_with_accuracy_control(ov_model,
                                                          calibration_dataset=calibration_dataset,
                                                          validation_dataset=validation_dataset,
                                                          validation_fn=validate,
                                                          max_drop=0.01)
    # save Model
    serialize(quantized_model, OUTPUT_MODEL_PATH)


if __name__ == "__main__":
    accuracy_quantization()
    # basic_quantization()
