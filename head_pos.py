#!/usr/bin/env python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read Image
def get_pose(ponts):
    size = (640, 480)
    # 2D image points. If you change the image, you need to change vector
    image_points = np.array([
        ponts[54],  # Nose tip
        ponts[16],  # Chin
        ponts[60],  # Left eye left corner
        ponts[72],  # Right eye right corne
        ponts[76],  # Left Mouth corner
        ponts[82]  # Right mouth corner
    ], dtype="double")
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = \
        cv2.solvePnP(model_points,
                     image_points,
                     camera_matrix,
                     dist_coeffs,
                     flags=cv2.SOLVEPNP_ITERATIVE) #

    (nose_end_point2D, jacobian) = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)]),
        rotation_vector,
        translation_vector,
        camera_matrix,
        dist_coeffs)
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    return p1,p2,rotation_vector


def drow_box(img, cnt):
    rect_box = cv2.boundingRect(cnt)
    rotated_box = cv2.minAreaRect(cnt)

    cv2.rectangle(img, (rect_box[0], rect_box[1]), (rect_box[0] + rect_box[2], rect_box[1] + rect_box[3]), (0, 255, 0), 2)

    box = cv2.boxPoints(rotated_box)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()

    return img, rotated_box, box

def crop1(img, cnt):
    horizon = True

    img, rotated_box, _ = drow_box(img, cnt)

    center, size, angle = rotated_box[0], rotated_box[1], rotated_box[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    print(angle)

    if horizon:
        if size[0] < size[1]:
            angle -= 270
            w = size[1]
            h = size[0]
        else:
            w = size[0]
            h = size[1]
        size = (w, h)

    height, width = img.shape[0], img.shape[1]

    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop


def return_cnt(data):
    ""
    left = []
    right = []
    top = []
    down = []
    tan = (right[1] - left[1])/(right[0] - left[0])
    arcthan = -1/tan

