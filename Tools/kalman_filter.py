import cv2
import numpy as np


class Kalman:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)  # 四个输入，需要预测两个
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)  # 传递矩阵
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32) * 0.02  # 噪声
        self.last_measurement = None
        self.last_prediction = None

    def Position_Predict(self, x, y):
        """
        :param x: 输入 x 坐标
        :param y: 输入 y 坐标
        :return:  返回预测的坐标 x, y 无打包
        """
        # 设置为全局变量
        measurement = np.array([[x], [y]], np.float32)
        # 第一次实际测量
        if self.last_measurement is None:
            self.kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)
            self.kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
            prediction = measurement
        # 不是第一次则进行预测
        else:
            self.kalman.correct(measurement)
            prediction = self.kalman.predict()
        # 进行迭代
        self.last_prediction = prediction.copy()
        self.last_measurement = measurement
        return prediction[:2][0], prediction[:2][1]

    def clean(self):
        self.last_measurement = None


"""
https://blog.csdn.net/weixin_55737425/article/details/124560990
"""
