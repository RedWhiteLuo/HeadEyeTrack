import cv2
import numpy as np
import torch
import win32gui
import win32ui
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Sequential, ReLU, Linear, LeakyReLU, Dropout
from torch.utils.tensorboard import SummaryWriter
from win32api import GetSystemMetrics
from FaceLandmark.core.facer import FaceAna
from Tools.dataloader import EpochDataLoader
from Tools.kalman_filter import Kalman
from facelandmark import trim_eye_img


def draw_rect(coords):
    dcObj.Rectangle((coords[0] - 3, coords[1] - 3, coords[0] + 3, coords[1] + 3))
    win32gui.InvalidateRect(hwnd, monitor, True)  # Refresh the entire monitor


def get_eye_img():
    ret, image = vide_capture.read()
    if ret:
        result = facer.run(image)
        for face_index in range(len(result)):
            face_kp, face_kp_score = result[face_index]['kps'], result[face_index]['scores']
            eye_img = trim_eye_img(image, face_kp)
            cv2.imshow('eye_img', eye_img)
            cv2.waitKey(1)
            eye_img = eye_img.transpose((2, 0, 1))
            eye_img = eye_img / 255  # [C, H, W] format
            eye_img = eye_img[None]
            return eye_img


class EyeTrackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            # in-> [N, 3, 32, 128]
            Conv2d(3, 20, kernel_size=(5, 5), padding=2),
            LeakyReLU(),
            #   -> [N, 20, 32, 128]
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            #   -> [N, 20, 16, 64]
            Conv2d(20, 20, kernel_size=(5, 5), padding=2),
            LeakyReLU(),
            Conv2d(20, 10, kernel_size=(3, 3), padding=1),
            ReLU(),
            #   -> [N, 10, 16, 64]
            Flatten(1, 3),
            Dropout(0.1),
            #   -> [N, 20480]
            Linear(10240, 2)
            # out-> [N, 2]
        )

    def forward(self, x):
        return self.model(x)


def train():
    train_img, train_coords = EpochDataLoader('E:/AI_Dataset/0Project/HeadEyeTrack', batch_size=512)
    batch_num = train_img.size()[0]
    learn_step, epoch_num, trained_batch_num = 0.01, 500, 0
    model = EyeTrackModel().to(device).train()
    loss = torch.nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=learn_step)
    writer = SummaryWriter('../train_logs')
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
        if epoch // 5 == 0:
            torch.save(model, "../model/ET-" + str(epoch) + ".pt")
    # save model
    torch.save(model, "../model/ET-last.pt")
    writer.close()
    print("[SUCCEED!] model saved!")


def run():
    km_filter = Kalman()
    model = torch.load("../model/ET-last.pt").eval()
    torch.no_grad()
    while True:
        # img = get_eye_img()
        img = cv2.imread("E:/AI_Dataset/0Project/HeadEyeTrack/0_1419_23.png")
        if img is not None:
            img = np.array(img)
            img = torch.from_numpy(img).float()
            img = img.to(device)
            outputs_ = model(img)
            x = int(outputs_[0][0] * 1920)
            y = int(outputs_[0][1] * 1080)
            (x, y) = km_filter.Position_Predict(x, y)
            draw_rect([int(x), int(y)])
        else:
            print("[ERROR] something wrong when trying to get eye img")


if __name__ == '__main__':
    # draw rectangle on screen
    dc = win32gui.GetDC(0)
    dcObj = win32ui.CreateDCFromHandle(dc)
    hwnd = win32gui.WindowFromPoint((0, 0))
    monitor = (0, 0, GetSystemMetrics(0), GetSystemMetrics(1))
    # initialize
    facer = FaceAna()
    vide_capture = cv2.VideoCapture(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # train or eval
    train()

"""benchmark_app -m ET-last-INT8.xml -d CPU -api async"""
