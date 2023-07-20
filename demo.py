import time
import cv2
from Skps.core.api.facer import FaceAna
from head_pos import get_pose

facer = FaceAna()
vide_capture = cv2.VideoCapture(0)

while True:
    ret, image = vide_capture.read()
    image = cv2.flip(image, 1)
    if ret:
        img_show = image.copy()
        
        star = time.time()
        result = facer.run(image)
        duration = time.time() - star
        
        fps = 1 / duration if duration else 0
        cv2.putText(img_show, "FPS: " + "{:7.2f}".format(fps), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0))

        for face_index in range(len(result)):

            face_kp, face_kp_score, index = result[face_index]['kps'], result[face_index]['scores'], 0
            for landmarks_index in range(face_kp.shape[0]):
                index += 1
                kp_coords = face_kp[landmarks_index]
                kp_coords = [int(kp_coords[0]), int(kp_coords[1])]
                if index == 61:
                    eyeleft, eyeright, eyetop, eyebottom = kp_coords[0], kp_coords[0], kp_coords[1], kp_coords[1]
                if 60 < index < 77:
                    eyeleft = kp_coords[0] if kp_coords[0] < eyeleft else eyeleft
                    eyeright = kp_coords[0] if kp_coords[0] > eyeright else eyeright
                    eyetop = kp_coords[1] if kp_coords[1] < eyetop else eyetop
                    eyebottom = kp_coords[1] if kp_coords[1] > eyebottom else eyebottom
                if index == 77:
                    eye_img = image[eyetop - 5:eyebottom + 5, eyeleft - 10:eyeright + 10]
                    cv2.namedWindow('eyes', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.imshow("eyes", eye_img)
                    
                    p1, p2, vector = get_pose(face_kp)
                    cv2.line(img_show, p1, p2, (0, 0, 255), 3)
                    cv2.putText(img_show, str(vector), (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), thickness=2)
                cv2.putText(img_show, str(index), (int(kp_coords[0]), int(kp_coords[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (255, 255, 255),
                            thickness=1)
                score = face_kp_score[landmarks_index]
                # color = (255, 255, 255)
                if score > 0.85:
                    color = (255, 255, 255)
                else:
                    color = (0, 0, int(255 * score))
                cv2.circle(img_show, (int(kp_coords[0]), int(kp_coords[1])),
                           color=color, radius=1, thickness=2)
        cv2.imshow("capture", img_show)

        key = cv2.waitKey(1)
