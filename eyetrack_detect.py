import cv2
import numpy as np
import win32gui
import win32ui
from win32api import GetSystemMetrics
from EyeTrack.core.eye_track_model import *  # import device and model
from FaceLandmark.core.facer import FaceAna
from Tools.kalman_filter import Kalman
from Tools.xml_model_base import XMLEngine
from facelandmark import trim_eye_img

# init draw rectangle on screen
dc = win32gui.GetDC(0)
dcObj = win32ui.CreateDCFromHandle(dc)
hwnd = win32gui.WindowFromPoint((0, 0))
monitor = (0, 0, GetSystemMetrics(0), GetSystemMetrics(1))


def draw_rect(coords):
    dcObj.Rectangle((coords[0] - 3, coords[1] - 3, coords[0] + 3, coords[1] + 3))
    win32gui.InvalidateRect(hwnd, monitor, True)  # Refresh the entire monitor


def get_eye_img():
    ret, image = vide_capture.read()
    if ret:
        result = facer.run(image)
        eye_img = None
        for face_index in range(len(result)):
            face_kp, face_kp_score = result[face_index]['kps'], result[face_index]['scores']
            eye_img = trim_eye_img(image, face_kp)
        if eye_img is None:
            print("[ERROR] | No faces are detected")
        return eye_img
    else:
        print("[ERROR] | CANT GET IMG FORM THE GIVEN CAMERA")


def eye_track_detect(path: str, single_img=None):
    km_filter = Kalman()
    if path.split(".")[-1] == "pt":
        model = torch.load(path).eval()
        while True:
            img = cv2.imread(single_img) if single_img is not None else get_eye_img()
            img = safe_img if img is None else img
            img = cv2.dnn.blobFromImage(img, 1 / 255.0, swapRB=True)
            img = torch.from_numpy(img).float().to(device)
            # infer (sync)
            outputs = model(img)
            # De-normalization and apply kalman filter
            x, y = int(outputs[0][0] * 1920), int(outputs[0][1] * 1080)
            x, y = km_filter.Position_Predict(x, y)
            # print
            print("[INFO]: coords: ", x, y)
            draw_rect([int(x), int(y)])

    elif path.split(".")[-1] == "xml":
        img = cv2.imread(single_img) if single_img is not None else get_eye_img()
        img = safe_img if img is None else img
        img = cv2.dnn.blobFromImage(img, 1 / 255.0)
        xml_engin(img)
        while True:
            img = cv2.imread(single_img) if single_img else get_eye_img()
            img = safe_img if img is None else img
            cv2.imshow("img", img)
            cv2.waitKey(1)
            img = cv2.dnn.blobFromImage(img, 1 / 255.0, swapRB=True)
            # get output and prepare start next infer
            outputs = xml_engin.get()
            xml_engin(img)
            # De-normalization and apply kalman filter
            x, y = int(outputs[0][0] * 1920), int(outputs[0][1] * 1080)
            x, y = km_filter.Position_Predict(x, y)
            draw_rect([int(x), int(y)])


if __name__ == '__main__':
    cam_id = 0
    model_path = "EyeTrack/model/ET-last_ACC_INT8.xml"
    singe_img_path = "E:/AI_Dataset/0Project/HeadEyeTrack/0_1419_23.png"
    # initialize
    torch.no_grad()
    facer = FaceAna()
    vide_capture = cv2.VideoCapture(cam_id)
    xml_engin = XMLEngine(model_path)   # async
    safe_img = cv2.imread("EyeTrack/core/safe_img.png")
    # train or eval
    eye_track_detect(model_path)
    # eye_track_detect(model_path, single_img=singe_img_path)

"""benchmark_app -m ET-last-INT8.xml -d CPU -api async"""
