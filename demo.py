import cv2
import time
import pyautogui

from Skps.core.api.facer import FaceAna
from head_face_vector import calculate_face_vector

facer = FaceAna()
vide_capture = cv2.VideoCapture(0)
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

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
            vector_2d = [vector_p2[i] - vector_p1[i] for i in range(2)]
            pyautogui.moveTo(vector_2d[0]*8 + 960, vector_2d[1]*8 + 540)
            # process key points data
            """for landmarks_index in range(face_kp.shape[0]):
                kp_coords = face_kp[landmarks_index]
                kp_score, kp_coords = face_kp_score[landmarks_index], [int(kp_coords[0]), int(kp_coords[1])]
                # mark key points in img and define color depend on score
                color = (0, 255, 0) if kp_score > 0.85 else (0, 0, int(255 * kp_score))
                cv2.circle(image, (int(kp_coords[0]), int(kp_coords[1])), color=color, radius=2, thickness=1)"""
        # show img
        cv2.imshow("capture", image)
        cv2.waitKey(1)
