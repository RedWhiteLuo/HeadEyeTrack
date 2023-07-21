import cv2
import time

import keyboard
import numpy as np
import pyautogui
from Skps.core.api.facer import FaceAna
from head_face_vector import calculate_face_vector

facer = FaceAna()
vide_capture = cv2.VideoCapture(0)
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False
saved_img_index = 0
global_eye_img, global_cursor_position = 0, 0


def save_img_and_coords(img, coords):
    global saved_img_index
    save_path = './dataset/' + '%d_%d_%d.png' % (saved_img_index, coords[0], coords[1])
    cv2.imwrite(save_path, img)
    saved_img_index += 1


keyboard.add_hotkey('w', save_img_and_coords, args=(global_eye_img, global_cursor_position), suppress=True)
if __name__ == '__main__':
    while True:
        ret, image = vide_capture.read()
        image = cv2.flip(image, 1)
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
                # show face vector
                vector_p1, vector_p2, vector = calculate_face_vector(face_kp)
                cv2.line(image, vector_p1, vector_p2, (0, 0, 255), 3)
                cv2.putText(image, str(vector), (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                # move mouse
                """vector_2d = [vector_p2[i] - vector_p1[i] for i in range(2)]
                pyautogui.moveTo(vector_2d[0]*8 + 960, vector_2d[1]*8 + 540)"""
                # process key points data
                """for landmarks_index in range(face_kp.shape[0]):
                    kp_coords = face_kp[landmarks_index]
                    kp_score, kp_coords = face_kp_score[landmarks_index], [int(kp_coords[0]), int(kp_coords[1])]
                    # mark key points in img and define color depend on score
                    color = (0, 255, 0) if kp_score > 0.85 else (0, 0, int(255 * kp_score))
                    cv2.circle(image, (kp_coords[0], kp_coords[1]), color=color, radius=2, thickness=1)"""
                # left / right eye coords -> eye_img
                l_l, l_r, l_t, l_b = face_kp[60][0], face_kp[64][0], face_kp[62][1], face_kp[66][1]
                r_l, r_r, r_t, r_b = face_kp[68][0], face_kp[72][0], face_kp[70][1], face_kp[74][1]
                left_eye_img = image[int(l_t):int(l_b), int(l_l):int(l_r)]
                right_eye_img = image[int(r_t):int(r_b), int(r_l):int(r_r)]
                left_eye_img = cv2.resize(left_eye_img, (64, 32), interpolation=cv2.INTER_AREA)
                right_eye_img = cv2.resize(right_eye_img, (64, 32), interpolation=cv2.INTER_AREA)
                eye_img = np.concatenate((left_eye_img, right_eye_img), axis=1)
                cv2.imshow('eye_img', eye_img)
                cv2.waitKey(1)
                # save picture and coords
                cursor_position = pyautogui.position()  # two-integer tuple
                #
            # show img
            cv2.imshow("capture", image)
            cv2.waitKey(1)
