import cv2
import torch
from openvino.runtime import Core, Tensor


class XMLEngine:
    def __init__(self, xml_model_path: str, device=None):
        self.core = Core()
        device = device if device else "AUTO"
        self.net = self.core.compile_model(xml_model_path, device_name=device)
        self.input_node, self.output_node = self.net.inputs[0], self.net.outputs[0]
        self.infer = self.net.create_infer_request()

    def __call__(self, frame_curr_tensor):
        infer = self.infer
        frame_curr_tensor = Tensor(frame_curr_tensor)
        infer.set_tensor(self.input_node, frame_curr_tensor)
        infer.start_async()
        return 0

    def get(self):
        self.infer.wait()
        infer_result = self.infer.get_tensor(self.output_node)
        return infer_result.data


if __name__ == "__main__":
    model_path = "../EyeTrack/model/ET-last_BASIC_INT8.xml"
    img_path = "E:/AI_Dataset/0Project/HeadEyeTrack/0_1419_23.png"
    engin = XMLEngine(model_path)
    img = cv2.imread(img_path)
    # Normalization + Swap RB + Layout from HWC to NCHW
    img = cv2.dnn.blobFromImage(img, 1 / 255.0, swapRB=True)
    print("[DEBUG-3ADS]", img.shape, type(img))
    engin(img)
    res = engin.get()
    print(res)


"""
BASIC_INT8:
[[0.41351908 0.3843597 ]]
ACC_INT8:
[[0.41351908 0.3843597 ]]
FP32:
[[0.41285115 0.38399538]]
"""