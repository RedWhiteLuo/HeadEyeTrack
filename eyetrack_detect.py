import cv2
from Tools.kalman_filter import Kalman
from Tools.process_data import trim_eye_img


def get_eye_img(vide_capture, facer):
    """
    :param vide_capture:  vide_capture = cv2.VideoCapture(cam_id)
    :param facer: facer = FaceAna()
    :return: trim_eye_img
    """
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


def eye_track_detect(path: str, single_img=None, if_kalman=False):
    vide_capture, facer = 0, 0
    km_filter = Kalman()
    if single_img is None:
        from FaceLandmark.core.facer import FaceAna
        facer = FaceAna()
        vide_capture = cv2.VideoCapture(CAM_ID)
        from Tools.process_data import trim_eye_img
    if path.split(".")[-1] == "pt":
        import torch
        torch.no_grad()
        model = torch.load(path).eval()
        while True:
            img = cv2.imread(single_img) if single_img is not None else get_eye_img(vide_capture, facer)
            if img is not None:
                img = cv2.dnn.blobFromImage(img, 1 / 255.0, swapRB=True)
                img = torch.from_numpy(img).float().to(device)
                # infer (sync)
                outputs = model(img)
                # De-normalization and apply kalman filter
                x, y = int(outputs[0][0] * 1920), int(outputs[0][1] * 1080)
                (x, y) = km_filter.Position_Predict(x, y) if if_kalman else (x, y)
                # print
                print("[INFO]: coords: ", x, y)

    elif path.split(".")[-1] == "xml":
        from Tools.xml_model_base import XMLEngine
        img = cv2.imread(single_img) if single_img is not None else get_eye_img(vide_capture, facer)
        if img is not None:
            img = cv2.dnn.blobFromImage(img, 1 / 255.0)
            xml_engin = XMLEngine(model_path)  # async
            xml_engin(img)
            while True:
                img = cv2.imread(single_img) if single_img else get_eye_img(vide_capture, facer)
                if img is not None:
                    cv2.imshow("img", img)
                    cv2.waitKey(1)
                    img = cv2.dnn.blobFromImage(img, 1 / 255.0, swapRB=True)
                    # get output and prepare start next infer
                    outputs = xml_engin.get()
                    xml_engin(img)
                    # De-normalization and apply kalman filter
                    x, y = int(outputs[0][0] * 1920), int(outputs[0][1] * 1080)
                    (x, y) = km_filter.Position_Predict(x, y) if if_kalman else (x, y)
                    print([int(x), int(y)])


if __name__ == '__main__':
    CAM_ID = 1
    model_path = "./EyeTrack/model/ET-last.pt"
    # if the model is pt model
    if model_path.split(".")[-1] == "pt":
        from EyeTrack.core.eye_track_model import EyeTrackModel, device
        print(EyeTrackModel())
    # read img
    singe_img_path = "./dataset/val/0_41_24.png"
    # train or eval
    # eye_track_detect(model_path)
    eye_track_detect(model_path, single_img=singe_img_path, if_kalman=False)

"""benchmark_app -m ET-last-INT8.xml -d CPU -api async"""
