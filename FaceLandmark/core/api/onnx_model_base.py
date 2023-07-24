import onnxruntime as rt


#
class ONNXEngine:
    def __init__(self, onnx_model_path, device='cuda'):
        self.device = device
        providers = ['CPUExecutionProvider']
        if 'cuda' in self.device:
            providers = ['CUDAExecutionProvider']
        self.session = rt.InferenceSession(onnx_model_path, providers=providers)

    def __call__(self, data):
        # support 1 input and 1 output for now
        # Inference
        y_onnx = self.session.run([], {self.session.get_inputs()[0].name: data})
        return y_onnx
