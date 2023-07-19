import argparse
import time

import cv2
import numpy as np
from Skps.core.api.facer import FaceAna


from head_pos import get_pose


def video(video_path_or_cam):
    facer = FaceAna()
    vide_capture = cv2.VideoCapture(video_path_or_cam)

    while True:
        ret, image = vide_capture.read()
        image = cv2.flip(image, 1)
        if ret:
            pattern = np.zeros_like(image)

            img_show = image.copy()
            star = time.time()
            result = facer.run(image)

            duration = time.time() - star
            # print('one image cost %f s'%(duration))
            duration = duration if duration else 1
            fps = 1 / duration
            cv2.putText(img_show, "FPS: " + "{:7.2f}".format(fps), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)

            for face_index in range(len(result)):
                # head pose need develop
                # reprojected, euler_angle=get_head_pose(landmarks[face_index],img_show)

                cur_face_kps = result[face_index]['kps']
                cur_face_kps_score = result[face_index]['scores']
                index = 0
                for landmarks_index in range(cur_face_kps.shape[0]):
                    index += 1
                    x_y = cur_face_kps[landmarks_index]
                    x_y = [int(x_y[0]), int(x_y[1])]
                    if index == 61:
                        eyeleft, eyeright, eyetop, eyebottom = x_y[0], x_y[0], x_y[1], x_y[1]
                    if 60 < index < 77:
                        eyeleft = x_y[0] if x_y[0] < eyeleft else eyeleft
                        eyeright = x_y[0] if x_y[0] > eyeright else eyeright
                        eyetop = x_y[1] if x_y[1] < eyetop else eyetop
                        eyebottom = x_y[1] if x_y[1] > eyebottom else eyebottom
                    if index == 77:
                        eye_img = image[eyetop - 5:eyebottom + 5, eyeleft - 10:eyeright + 10]
                        cv2.namedWindow('eyes', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                        cv2.imshow("eyes", eye_img)
                        cv2.rectangle(img_show, (eyeleft, eyetop), (eyeright, eyebottom), (255, 0, 255), -1)
                        # print(eyeleft,eyetop,eyeright,eyebottom)
                        p1, p2, vector = get_pose(cur_face_kps)
                        cv2.line(img_show, p1, p2, (0, 0, 255), 3)
                        cv2.putText(img_show, str(vector), (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 255), thickness=2)
                    cv2.putText(img_show, str(index), (int(x_y[0]), int(x_y[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                (255, 255, 255),
                                thickness=1)
                    score = cur_face_kps_score[landmarks_index]
                    # color = (255, 255, 255)
                    if score > 0.85:
                        color = (255, 255, 255)
                    else:
                        color = (0, 0, 255)
                    cv2.circle(img_show, (int(x_y[0]), int(x_y[1])),
                               color=color, radius=1, thickness=2)
            cv2.imshow("capture", img_show)

            key = cv2.waitKey(1)
            if key == ord('q'):
                return


def build_argparse():
    parser = argparse.ArgumentParser(description='Start train.')
    parser.add_argument('--video', dest='video', type=str, default=None, \
                        help='the camera id (default: 0)')
    parser.add_argument('--cam_id', dest='cam_id', type=int, default=0, \
                        help='the camera to use')
    parser.add_argument('--img_dir', dest='img_dir', type=str, default=None, \
                        help='the images dir to use')

    parser.add_argument('--mask', dest='mask', type=bool, default=False, \
                        help='mask the face or not')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    video(0)
