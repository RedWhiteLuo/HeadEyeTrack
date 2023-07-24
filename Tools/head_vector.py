#!/usr/bin/env python
import cv2
import numpy as np

from Tools.kalman_filter import Kalman as KM

rotation_vector_filter = [KM(), KM(), KM()]
vector_2d_filter = KM()


def calculate_face_vector(points):
    size = (480, 640)
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    # 2D image points. If you change the image, you need to change vector
    image_points = np.array([
        points[54],  # Nose tip
        points[16],  # Chin
        points[60],  # Left eye's left corner
        points[72],  # Right eye's right corne
        points[76],  # Left Mouth corner
        points[82]  # Right mouth corner
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye's left corner
        (225.0, 170.0, -135.0),  # Right eye's right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # Camera internals
    focal_length = size[0]
    center = (size[0] / 2, size[1] / 2)
    camera_matrix = \
        np.array([[focal_length, 0, center[1]],
                  [0, focal_length, center[0]],
                  [0, 0, 1]], dtype="double")

    (success, rotation_vector, translation_vector) = \
        cv2.solvePnP(model_points,
                     image_points,
                     camera_matrix,
                     dist_coeffs,
                     flags=cv2.SOLVEPNP_ITERATIVE)

    "Apply Kalman Filter to rotation_vector"
    # for i in range(3):
    #    rotation_vector[i][0] = rotation_vector_filter[i].Position_Predict(rotation_vector[i][0], 0)[0]

    (nose_end_point2D, jacobian) = \
        cv2.projectPoints(np.array([(0.0, 0.0, 800.0)]),
                          rotation_vector,
                          translation_vector,
                          camera_matrix,
                          dist_coeffs)

    p1 = [int(image_points[0][0]), int(image_points[0][1])]  # nose point
    p2 = [int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])]
    # KM filter
    rela_p = [p2[i] - p1[i] for i in range(2)]
    rela_p = vector_2d_filter.Position_Predict(rela_p[0], rela_p[1])
    p2_filter = [int(p1[i] + rela_p[i]) for i in range(2)]
    # limit data len
    rotation_vector[0] += 3.2
    rotation_vector = [round(rotation_vector[i][0], 8) for i in range(3)]
    return p1, p2_filter, rotation_vector


"""
https://blog.csdn.net/weixin_41010198/article/details/116028666
"""
