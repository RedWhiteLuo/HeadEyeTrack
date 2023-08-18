import os.path
import pathlib
import cv2
import numpy as np
import yaml
from FaceLandmark.core.api.face_detector import FaceDetector
from FaceLandmark.core.api.face_landmark import FaceLandmark
from FaceLandmark.core.smoother.lk import GroupTrack, EmaFilter
from FaceLandmark.logger import logger


def get_cfg():
    root_path = pathlib.Path(__file__).resolve().parents[2]
    cfg_path = os.path.join(root_path, 'config.yml')
    with open(cfg_path, encoding="UTF-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


class FaceAna:

    def __init__(self):

        cfg = get_cfg()

        self.face_detector = FaceDetector(cfg['Detect'])
        self.face_landmark = FaceLandmark(cfg['Keypoints'])
        self.trace = GroupTrack(cfg['Trace'])
        # another thread should run detector in a slow way and update the track_box
        self.track_box = None
        self.previous_image = None
        self.previous_box = None
        self.diff_thres = 5
        self.top_k = cfg['Detect']['topk']
        self.min_face = cfg['Detect']['min_face']
        self.iou_thres = cfg['Trace']['iou_thres']
        self.alpha = cfg['Trace']['smooth_box']
        self.filter = EmaFilter(self.alpha)
        logger.info('model init done!')

    def run(self, image: object):
        """
        Args:
            image:  get [H, W, C]
        Returns:
            dict [:98]{kps : [x,y], scores : num }
        """
        # run detector
        if self.diff_frames(self.previous_image, image):  # if same
            boxes = self.face_detector(image)  # get the bounding box of each people
            self.previous_image = image
            boxes = self.judge_boxs(self.track_box, boxes)  # return the bounding box after EmaFilter
            self.trace.previous_landmarks_set = None
        else:
            boxes = self.track_box
            self.previous_image = image

        boxes = self.sort_and_filter(boxes)
        boxes_return = np.array(boxes)
        landmarks, states = self.face_landmark(image, boxes)

        # refine the landmark
        landmarks = self.trace.calculate(image, landmarks)

        # refine the bboxes
        track = []
        for i in range(landmarks.shape[0]):
            track.append([np.min(landmarks[i][:, 0]),
                          np.min(landmarks[i][:, 1]),
                          np.max(landmarks[i][:, 0]),
                          np.max(landmarks[i][:, 1])])
        tmp_box = np.array(track)
        self.track_box = self.judge_boxs(boxes_return, tmp_box)
        result = self.to_dict(self.track_box, landmarks, states)
        return result

    def to_dict(self, bboxes, kps, states):
        ans = []
        for i in range(len(bboxes)):
            one_res = {'box': bboxes[i], 'kps': kps[i], "scores": states[i]}
            ans.append(one_res)
        return ans

    def diff_frames(self, previous_frame, image):
        """
        diff value for two value,
        determin if to excute the detection
        :param previous_frame:  RGB_array
        :param image:           RGB_array
        :return:                True or False
        """
        if previous_frame is None:
            return True
        else:

            _diff = cv2.absdiff(previous_frame, image)
            diff = np.sum(_diff) / previous_frame.shape[0] / previous_frame.shape[1] / 3.

            if diff > self.diff_thres:
                return True
            else:
                return False

    def sort_and_filter(self, bboxes):
        """
        find the top_k max bboxes, and filter the small face
        :param bboxes:
        :return:
        """

        if len(bboxes) < 1:
            return []

        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        select_index = area > self.min_face  # filter face which is too small

        area = area[select_index]
        bboxes = bboxes[select_index, :]
        if bboxes.shape[0] > self.top_k:  # limit the number of face whill be marked keypoint
            picked = area.argsort()[-self.top_k:][::-1]  # return index that form the biggest face bbox to the smallest
            sorted_bboxes = [bboxes[x] for x in picked]
        else:
            sorted_bboxes = bboxes
        return np.array(sorted_bboxes)

    def judge_boxs(self, previous_bboxs, now_bboxs):
        """
        function used to calculate the tracking bboxes
        :param previous_bboxs:[[x1,y1,x2,y2],... ]
        :param now_bboxs: [[x1,y1,x2,y2],... ]
        :return:
        """

        def iou(rec1, rec2):

            # computing area of each rectangle
            S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the edge of intersect rectangle
            x1 = max(rec1[0], rec2[0])
            y1 = max(rec1[1], rec2[1])
            x2 = min(rec1[2], rec2[2])
            y2 = min(rec1[3], rec2[3])

            # judge if there is an intersect
            intersect = max(0, x2 - x1) * max(0, y2 - y1)

            return intersect / (sum_area - intersect)

        if previous_bboxs is None:
            return now_bboxs

        result = []
        for i in range(now_bboxs.shape[0]):
            contain = False
            for j in range(previous_bboxs.shape[0]):
                if iou(now_bboxs[i], previous_bboxs[j]) > self.iou_thres:
                    result.append(self.smooth(now_bboxs[i], previous_bboxs[j]))
                    contain = True
                    break
            if not contain:
                result.append(now_bboxs[i][0:4])

        return np.array(result)

    def smooth(self, now_box, previous_box):

        return self.filter(now_box[:4], previous_box[:4])

    def reset(self):
        """
        reset the previous info used foe tracking,

        :return:
        """
        self.track_box = None
        self.previous_image = None
        self.previous_box = None
