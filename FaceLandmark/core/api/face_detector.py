import pathlib
import time
import cv2
import numpy as np
import os
from openvino.runtime import Core
from FaceLandmark.logger.logger import logger


class FaceDetector:
    def __init__(self, cfg):
        root_path = pathlib.Path(__file__).resolve().parents[2]
        model_path_xml = os.path.join(root_path, cfg['model_path_xml'])
        self.model = Core().compile_model(model_path_xml, "GPU.0")
        self.output_node = self.model.outputs[0]
        self.input_size = cfg['input_shape']
        self.score_thrs = cfg['score_thrs']
        self.iou_thrs = cfg['iou_thrs']

    def __call__(self, image):
        """
        Args:
            image:  receive img color format: [BGR], size unlimited
        Returns:
            face detect bounding box XYXY-{in origin img} format
        """
        start_time = time.perf_counter()
        img_for_net, recover_info = self.preprocess(image)
        # infer
        output = self.model([img_for_net])[self.output_node]
        # convert infer result
        output = np.reshape(output, (15120, 16))
        output[:, :4] = self.xywh2xyxy(output[:, :4])
        bboxes = self.py_nms(output, self.iou_thrs, self.score_thrs)
        bboxes[:, :4] = self.scale_coords(bboxes[:, :4], recover_info)
        duration = time.perf_counter() - start_time
        logger.info('[Face] Total time: %.5fs' % duration)
        return bboxes

    def preprocess(self, image, color=(114, 114, 114)):
        origin_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # resize origin img to model input required size
        h, w, c = origin_img.shape
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        resize_img = cv2.resize(origin_img, (int(w * scale), int(h * scale)))
        # fill img if ratio not fit model
        h, w, c = resize_img.shape
        dh = (self.input_size[0] - h) / 2
        dw = (self.input_size[1] - w) / 2
        " make sure round(0.5) + round(0.5) = 1 "
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        # convert img format
        img = img.transpose(2, 0, 1).astype(np.float16)     # convert to [C,W,H]
        img /= 255.0
        img = np.expand_dims(img, axis=0)

        return img, [scale, left, top]

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

        return y

    def scale_coords(self, bbox, revocer_info):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        [scale, dx, dy] = revocer_info

        bbox[:, 0] -= dx
        bbox[:, 1] -= dy
        bbox[:, 2] -= dx
        bbox[:, 3] -= dy

        bbox /= scale

        return bbox

    def py_nms(self, bboxes, iou_thres, score_thres):

        upper_thres = np.where(bboxes[:, 4] > score_thres)[0]
        bboxes = bboxes[upper_thres]

        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        order = np.argsort(bboxes[:, 4])[::-1]

        keep = []
        while order.shape[0] > 0:
            cur = order[0]
            keep.append(cur)
            area = (bboxes[cur, 2] - bboxes[cur, 0]) * (bboxes[cur, 3] - bboxes[cur, 1])

            x1_reain = x1[order[1:]]
            y1_reain = y1[order[1:]]
            x2_reain = x2[order[1:]]
            y2_reain = y2[order[1:]]

            xx1 = np.maximum(bboxes[cur, 0], x1_reain)
            yy1 = np.maximum(bboxes[cur, 1], y1_reain)
            xx2 = np.minimum(bboxes[cur, 2], x2_reain)
            yy2 = np.minimum(bboxes[cur, 3], y2_reain)

            intersection = np.maximum(0, yy2 - yy1) * np.maximum(0, xx2 - xx1)
            iou = intersection / (area + (y2_reain - y1_reain) * (x2_reain - x1_reain) - intersection)

            # keep the low iou
            low_iou_position = np.where(iou < iou_thres)[0]
            order = order[low_iou_position + 1]

        return bboxes[keep]

