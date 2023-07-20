import cv2
import time
from Skps.core.api.facer import FaceAna
from head_pos import calculate_face_vector

facer = FaceAna()
vide_capture = cv2.VideoCapture(0)

while True:
    ret, image = vide_capture.read()
    image = cv2.flip(image, 1)
    if ret:
        star = time.perf_counter()
        result = facer.run(image)
        duration = time.perf_counter() - star

        fps = 1 / duration if duration else 0
        cv2.putText(image, "FPS: " + "{:7.2f}".format(fps), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0))
        t1 = time.perf_counter()
        for face_index in range(len(result)):

            face_kp, face_kp_score, index = result[face_index]['kps'], result[face_index]['scores'], 0

            vector_p1, vector_p2, vector = calculate_face_vector(face_kp)
            cv2.line(image, vector_p1, vector_p2, (0, 0, 255), 3)
            cv2.putText(image, str(vector), (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            for landmarks_index in range(face_kp.shape[0]):
                kp_score = face_kp_score[landmarks_index]
                kp_coords = face_kp[landmarks_index]
                kp_coords = [int(kp_coords[0]), int(kp_coords[1])]

                # color = (255, 255, 255)
                color = (0, 255, 0) if kp_score > 0.85 else (0, 0, int(255 * kp_score))
                cv2.circle(image, (int(kp_coords[0]), int(kp_coords[1])), color=color, radius=2, thickness=1)

        cv2.imshow("capture", image)
        cv2.waitKey(1)
        end = time.perf_counter()
        print(end - t1)
