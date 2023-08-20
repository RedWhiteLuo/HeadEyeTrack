import argparse
import cv2
import time
import pyautogui
from FaceLandmark.core.facer import FaceAna
from Tools.head_vector import calculate_face_vector
from Tools.process_data import trim_eye_img, save_img_and_coords

facer = FaceAna()
HEIGHT, WEIGHT = 1920, 1080
vide_capture = cv2.VideoCapture(1)
vide_capture.set(cv2.CAP_PROP_FRAME_WIDTH, HEIGHT)
vide_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, WEIGHT)
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False


def run(
        flip_img=False,
        show_vector=False,
        draw_points=False,
        show_img=False,
        show_eye=False,
        save_dataset=False,
):
    while True:
        saved_img_index = 0
        ret, image = vide_capture.read()
        image = cv2.flip(image, 1) if flip_img else image
        if ret:
            # start infer and count time
            star = time.perf_counter()
            result = facer.run(image)
            duration = time.perf_counter() - star
            # show fps in img
            fps = 1 / duration if duration else 0
            cv2.putText(image, "FPS: " + "{:7.2f}".format(fps), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0))
            # process per face key points data
            for face_index in range(len(result)):
                face_kp, face_kp_score = result[face_index]['kps'], result[face_index]['scores']
                eye_img = trim_eye_img(image, face_kp)
                # show face vector
                if show_vector:
                    vector_p1, vector_p2, vector = calculate_face_vector(face_kp, (HEIGHT, WEIGHT))
                    cv2.line(image, vector_p1, vector_p2, (0, 0, 255), 3)
                    # cv2.putText(image, str(vector), (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                # draw key points
                if draw_points:
                    for landmarks_index in range(face_kp.shape[0]):
                        kp_coords = face_kp[landmarks_index]
                        kp_score, kp_coords = face_kp_score[landmarks_index], [int(kp_coords[0]), int(kp_coords[1])]
                        # mark key points in img and define color depend on score
                        color = (0, 255, 0) if kp_score > 0.85 else (0, 0, 255)
                        cv2.circle(image, (kp_coords[0], kp_coords[1]), color=color, radius=2, thickness=2)
                # show eye img separately
                if show_eye:
                    cv2.imshow('eye_img', eye_img)
                    cv2.waitKey(1)
                # save picture and coords
                if save_dataset:
                    cursor_coords = pyautogui.position()
                    save_img_and_coords(eye_img, cursor_coords, saved_img_index)
                    saved_img_index += 1
            # show img
            if show_img or show_vector or draw_points:
                pic = cv2.resize(image, (960, 540), interpolation=cv2.INTER_CUBIC)
                cv2.imshow("capture", pic)
                cv2.waitKey(1)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flip-img', action='store_true', help='show the face vector')
    parser.add_argument('--show-vector', action='store_true', help='show the face vector')
    parser.add_argument('--draw-points', action='store_true', help='draw point in the img')
    parser.add_argument('--show-img', action='store_true', help='if show img')
    parser.add_argument('--show-eye', action='store_true', help='show eye img after concatenate')
    parser.add_argument('--save-dataset', action='store_true', help='creat train dataset')
    _opt = parser.parse_args()
    return _opt


if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))
