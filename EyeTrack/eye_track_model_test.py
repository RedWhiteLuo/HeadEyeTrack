import torch
from torch import nn
import cv2
import numpy as np
import pyautogui
from FaceLandmark.core.api.facer import FaceAna
from face_landmark_demo import return_boundary
from torch.nn import Conv2d, MaxPool2d, Flatten, Sequential, ReLU, Linear, LeakyReLU, Dropout
from torch.utils.tensorboard import SummaryWriter
from Tools.DataLoader import DL
from Tools.kalman_filter import Kalman
import win32gui, win32ui
from win32api import GetSystemMetrics


dc = win32gui.GetDC(0)
dcObj = win32ui.CreateDCFromHandle(dc)
hwnd = win32gui.WindowFromPoint((0, 0))
monitor = (0, 0, GetSystemMetrics(0), GetSystemMetrics(1))


def draw_rect(coords):
    dcObj.Rectangle((coords[0]-5, coords[1]-5, coords[0] + 5, coords[1] + 5))
    win32gui.InvalidateRect(hwnd, monitor, True)  # Refresh the entire monitor


facer = FaceAna()
vide_capture = cv2.VideoCapture(0)
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False
saved_img_index = 0
global_eye_img, global_cursor_position = 0, 0
kmf = Kalman()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_eye_img():
    ret, image = vide_capture.read()
    cv2.imshow('face_img', image)
    cv2.waitKey(1)
    image = cv2.flip(image, 1)
    if ret:
        result = facer.run(image)
        for face_index in range(len(result)):
            face_kp, face_kp_score = result[face_index]['kps'], result[face_index]['scores']
            l_l, l_r, l_t, l_b = return_boundary(face_kp[60:68])
            r_l, r_r, r_t, r_b = return_boundary(face_kp[68:76])
            left_eye_img = image[int(l_t):int(l_b), int(l_l):int(l_r)]
            right_eye_img = image[int(r_t):int(r_b), int(r_l):int(r_r)]
            left_eye_img = cv2.resize(left_eye_img, (64, 32), interpolation=cv2.INTER_AREA)
            right_eye_img = cv2.resize(right_eye_img, (64, 32), interpolation=cv2.INTER_AREA)
            eye_img = np.concatenate((left_eye_img, right_eye_img), axis=1)
            cv2.imshow('eye_img', eye_img)
            cv2.waitKey(1)
            eye_img = eye_img.transpose((2, 0, 1))
            eye_img = eye_img / 255  # [C, H, W] format
            eye_img = eye_img[None]
            return eye_img


class EyeTrackModelStruct(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO(prof.redwhite@gmail.com): change model structure
        self.model = Sequential(
            # in-> [N, 3, 32, 128]
            Conv2d(3, 20, kernel_size=(5, 5), padding=2),  # keep W H
            LeakyReLU(),
            #   -> [N, 20, 32, 128] #
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            #   -> [N, 20, 16, 64]  #
            Conv2d(20, 20, kernel_size=(5, 5), padding=2),  # keep W H
            LeakyReLU(),
            Conv2d(20, 10, kernel_size=(3, 3), padding=1),  # keep W H
            ReLU(),
            #   -> [N, 10, 16, 64]
            Flatten(1, 3),
            Dropout(0.1),
            #   -> [N, 20480]
            Linear(10240, 2)
            # out-> [N, 2]
        )

    def forward(self, x):
        x = self.model(x)
        return x


def train():
    train_img, train_coords = DL('E:/AI_Dataset/0Project/HeadEyeTrackt', batch_size=512)
    batch_num = train_img.size()[0]
    learn_step, epoch_num, trained_batch_num = 0.01, 500, 0
    model = EyeTrackModelStruct().to(device)
    model.train()
    loss = torch.nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=learn_step)
    writer = SummaryWriter('./logs')
    try:
        for epoch in range(epoch_num):
            for batch in range(batch_num):
                batch_img = train_img[batch].to(device)
                batch_coords = train_coords[batch].to(device)
                # infer and calculate loss
                outputs = model(batch_img)
                result_loss = loss(outputs, batch_coords)
                # reset grad and calculate grad then optim model
                optim.zero_grad()
                result_loss.backward()
                optim.step()
                # save loss and print info
                trained_batch_num += 1
                writer.add_scalar("loss", result_loss.item(), trained_batch_num)
                print(epoch + 1, trained_batch_num, result_loss.item())
    except KeyboardInterrupt:
        pass
    # save model
    torch.save(model, "ET.pt")
    writer.close()
    print("[Finish!] model saved!")


def run():
    # TODO(prof.redwhite@gmail.com): convert model to openvino format and infer
    model = torch.load("ET.pt")
    model.eval()
    torch.no_grad()
    while True:
        img = get_eye_img()
        img = np.array(img)
        img = torch.from_numpy(img).float()
        img = img.to(device)
        outputs_ = model(img)
        x = int(outputs_[0][0] * 1920)
        y = int(outputs_[0][1] * 1080)
        (x, y) = kmf.Position_Predict(x, y)
        draw_rect([int(x), int(y)])


if __name__ == '__main__':
    train()
