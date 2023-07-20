# -*-coding:utf-8-*-
import time
import cv2
import numpy as np
import pathlib
import os
from openvino.runtime import Core
from Skps.logger import logger


class FaceLandmark:

    def __init__(self, cfg):
        root_path = pathlib.Path(__file__).resolve().parents[2]
        model_path_xml = os.path.join(root_path, cfg['model_path_xml'])
        self.model = Core().compile_model(model_path_xml, "GPU.0")
        self.output_node = self.model.outputs
        self.min_face = 20
        self.keypoint_num = cfg['num_points']
        self.input_size = cfg['input_shape']
        self.extend = cfg['base_extend_range']

    # below are the method  run for one by one, will be deprecated in the future
    def __call__(self, img, bboxes):
        landmark_result, states_result = [], []

        t0 = time.perf_counter()
        for i, bbox in enumerate(bboxes):
            # process img before infer
            image_cropped, detail = self.preprocess(img, bbox)
            image_cropped = image_cropped.transpose((2, 0, 1)).astype(np.float16)
            image_cropped = image_cropped / 255.
            image_cropped = np.expand_dims(image_cropped, axis=0)
            # infer
            result = self.model([image_cropped])
            landmark, score = result[self.output_node[0]], result[self.output_node[1]]
            # convert infer result
            state = score.reshape(-1)
            landmark = np.array(landmark)[:98 * 2].reshape(-1, 2)
            landmark = self.post_process(landmark, detail)

            if landmark is not None:
                landmark_result.append(landmark)
                states_result.append(state)

        if len(bboxes) > 0:
            duration = time.perf_counter() - t0
            logger.info('[Keypoint] %.5fs per face' % (duration / len(bboxes)))

        return np.array(landmark_result), np.array(states_result)

    def preprocess(self, image, bbox):
        """
        Args:
            image:  RGB format
            bbox:   xyxy format
        Returns:
            resized face img
            [h-{origin face img}, w-{origin face img}, bbox[1]-{TOP_Y}, bbox[0]-{LEFT_X}, add-{img expand pixel}]
        """
        bbox_width = bbox[2] - bbox[0]      # img W
        bbox_height = bbox[3] - bbox[1]     # img H
        " filter face that too small "
        if bbox_width <= self.min_face or bbox_height <= self.min_face:
            return None, None
        # expand the img, make sure all bbox can convert to square and prevent index error
        add = int(max(bbox_width, bbox_height))
        border_img = cv2.copyMakeBorder(image, add, add, add, add, borderType=cv2.BORDER_CONSTANT)
        bbox += add     # adjust the bbox due to the img expand
        # expand the bounding box, defined in the config file
        face_width = (1 + 2 * self.extend[0]) * bbox_width
        center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
        # make the box as square
        bbox[0] = center[0] - face_width // 2
        bbox[1] = center[1] - face_width // 2
        bbox[2] = center[0] + face_width // 2
        bbox[3] = center[1] + face_width // 2
        # crop the face
        bbox = bbox.astype(np.int32)
        crop_image = border_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        # resize the face img to model require
        h, w, _ = crop_image.shape
        crop_image = cv2.resize(crop_image, (self.input_size[1], self.input_size[0]))
        # cv2.imshow('i am watching u * * %d' % i, crop_image)
        return crop_image, [h, w, bbox[1], bbox[0], add]

    def post_process(self, landmark, detail):
        """
        Args:
            landmark: landmark coords in face img
            detail:
            [h-{origin face img}, w-{origin face img}, bbox[1]-{TOP_Y}, bbox[0]-{LEFT_X}, add-{img expand pixel}]
        Returns:
            face landmark coords in origin img
        """
        landmark[:, 0] = landmark[:, 0] * detail[1] + detail[3] - detail[4]
        landmark[:, 1] = landmark[:, 1] * detail[0] + detail[2] - detail[4]

        return landmark
