import nncf
import torch
import openvino.tools.mo as mo
from openvino.runtime import Core, serialize  # serialize actually works
from EyeTrack.core.eye_track_model import EyeTrackModel
from Tools.dataloader import create_data_source
from Tools.process_data import transform_fn, transform_fn_with_annot, validate


def basic_quantization(input_model_path):
    # prepare required data
    data_source = create_data_source(path=DATASET_ROOT_PATH)
    nncf_calibration_dataset = nncf.Dataset(data_source, transform_fn)
    # set the parameter of how to quantize
    subset_size = 1000
    preset = nncf.QuantizationPreset.MIXED  # or you can choose: nncf.QuantizationPreset.MIXED
    # load model
    ov_model = Core().read_model(input_model_path)
    # perform quantize
    quantized_model = nncf.quantize(ov_model, nncf_calibration_dataset, preset=preset, subset_size=subset_size)
    # save model
    output_model_path = input_model_path.split(".")[0] + "_BASIC_INT8.xml"
    serialize(quantized_model, output_model_path)


def accuracy_quantization(input_model_path, max_drop):
    # prepare required data
    calibration_source = create_data_source(path=DATASET_ROOT_PATH, with_annot=False)
    validation_source = create_data_source(path=DATASET_ROOT_PATH, with_annot=True)
    calibration_dataset = nncf.Dataset(calibration_source, transform_fn)
    validation_dataset = nncf.Dataset(validation_source, transform_fn_with_annot)
    # load model
    xml_model = Core().read_model(input_model_path)
    # perform quantize
    quantized_model = nncf.quantize_with_accuracy_control(xml_model,
                                                          calibration_dataset=calibration_dataset,
                                                          validation_dataset=validation_dataset,
                                                          validation_fn=validate,
                                                          max_drop=max_drop)
    # save model
    output_model_path = xml_model_path.split(".")[0] + "_ACC_INT8.xml"
    serialize(quantized_model, output_model_path)


def export_onnx(model_path, if_fp16=False):
    """
    :param model_path: the path that will be converted
    :param if_fp16: if the output onnx model compressed to fp16
    :return: output xml model path
    """
    model = torch.load(model_path, map_location=torch.device('cpu')).eval()
    model_path = model_path.split(".")[0]
    dummy_input = torch.randn(1, 3, 32, 128, device='cpu')
    torch.onnx.export(model, dummy_input, model_path + ".onnx", export_params=True)
    model = mo.convert_model(model_path + ".onnx", compress_to_fp16=if_fp16)  # if_fp16=False, output = FP32
    serialize(model, model_path + ".xml")
    print(EyeTrackModel(), "\n[FINISHED] CONVERT DONE!")
    return model_path + ".xml"


if __name__ == "__main__":
    # convert pt model to onnx and xml format
    pt_model_path = 'model/ET-last.pt'
    xml_model_path = export_onnx(pt_model_path)
    #
    DATASET_ROOT_PATH = "E:/AI_Dataset/0Project/HeadEyeTrack/"
    # quantize model
    basic_quantization(xml_model_path)
    accuracy_quantization(xml_model_path, max_drop=0.01)