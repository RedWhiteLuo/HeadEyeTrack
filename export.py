import argparse
import torch
import openvino.tools.mo as mo
from openvino.runtime import serialize  # it actually works
from eyetrack import EyeTrackModelStruct


def run():
    model = torch.load('ET-last.pt', map_location=torch.device('cpu')).eval()
    print(model)
    dummy_input = torch.randn(1, 3, 32, 128, device='cpu')
    torch.onnx.export(model, dummy_input, "ET-last.onnx", export_params=True)
    model = mo.convert_model("ET-last.onnx", compress_to_fp16=False)    # FP32
    serialize(model, "ET-last-FP32.xml")
    print("[FINISHED] CONVERT DONE!")


if __name__ == '__main__':
    run()
