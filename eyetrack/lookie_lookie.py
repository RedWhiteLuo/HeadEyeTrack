import cv2
import torch
from torch import nn
import cv2
import time
import numpy as np
import pyautogui
from Skps.core.api.facer import FaceAna

facer = FaceAna()
vide_capture = cv2.VideoCapture(0)
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False
saved_img_index = 0
global_eye_img, global_cursor_position = 0, 0
from torch.nn import Conv2d, MaxPool2d, Flatten, Sequential, Dropout, Conv1d, Sigmoid, ReLU, Linear, MaxUnpool2d
from torch.utils.tensorboard import SummaryWriter
from DataLoader import DL

train_img, train_coords = DL('../dataset', batch_size=1024)
batch_num = train_img.size()[0]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_eye_img():
    ret, image = vide_capture.read()
    image = cv2.flip(image, 1)
    if ret:
        result = facer.run(image)
        for face_index in range(len(result)):
            face_kp, face_kp_score = result[face_index]['kps'], result[face_index]['scores']
            l_l, l_r, l_t, l_b = face_kp[60][0] - 2, face_kp[64][0] + 2, face_kp[62][1] - 2, face_kp[66][1] + 2
            r_l, r_r, r_t, r_b = face_kp[68][0] - 2, face_kp[72][0] + 2, face_kp[70][1] - 2, face_kp[74][1] + 2
            left_eye_img = image[int(l_t):int(l_b), int(l_l):int(l_r)]
            right_eye_img = image[int(r_t):int(r_b), int(r_l):int(r_r)]
            left_eye_img = cv2.resize(left_eye_img, (64, 32), interpolation=cv2.INTER_AREA)
            right_eye_img = cv2.resize(right_eye_img, (64, 32), interpolation=cv2.INTER_AREA)
            eye_img = np.concatenate((left_eye_img, right_eye_img), axis=1)
            eye_img = eye_img.transpose((2, 0, 1))
            eye_img = eye_img / 255  # [C, H, W] format
            eye_img = eye_img[None]
            return eye_img


class EyeTrack(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            # in-> [N, 3, 32, 128]
            Conv2d(3, 20, kernel_size=(5, 5), padding=2),  # keep W H
            #   -> [N, 20, 32, 128] #
            ReLU(),
            MaxPool2d(kernel_size=(2, 2)),
            #   -> [N, 20, 16, 64]  #
            Conv2d(20, 20, kernel_size=(5, 5), padding=2),  # keep W H
            ReLU(),
            #   -> [N, 20, 16, 64]
            Flatten(1, 3),
            #   -> [N, 20480]
            Linear(20480, 2)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def train():
    learn_step = 0.01
    epoch_num = 10000
    model = EyeTrack().to(device)
    loss = torch.nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=learn_step)
    writer = SummaryWriter('./logs')
    model.train()  # 模型在训练状态
    trained_batch_num = 0
    try:
        for epoch in range(epoch_num):
            for batch in range(batch_num):
                batch_img = train_img[batch].to(device)
                batch_coords = train_coords[batch].to(device)
                # model
                outputs = model(batch_img)  # forward infer
                result_loss = loss(outputs, batch_coords)
                optim.zero_grad()
                result_loss.backward()
                optim.step()
                trained_batch_num += 1
                writer.add_scalar("loss", result_loss.item(), trained_batch_num)
                print(epoch + 1, trained_batch_num, result_loss.item())
    except KeyboardInterrupt:
        pass
    torch.save(model, "ET.pt")
    print("model saved!")
    writer.close()


def run():
    model = torch.load("ET.pt")
    model.eval()  # 在验证状态
    right_number = 0
    torch.no_grad()
    while True:
        img = get_eye_img()
        img = np.array(img)
        img = torch.from_numpy(img).float()
        img = img.to(device)
        outputs_ = model(img)
        x = int(outputs_[0][0] * 1920)
        y = int(outputs_[0][1] * 1080)
        pyautogui.moveTo(x, y)
        print(x, y)


if __name__ == '__main__':
    run()
    # 在测试集上面的效果
    """mymodule.eval()  # 在验证状态
    test_total_loss = 0
    right_number = 0
    with torch.no_grad():  # 验证的部分，不是训练所以不要带入梯度
        for test_data in test_dataloader:
            imgs, label = test_data

            imgs = imgs.to(device)
            label = label.to(device)

            outputs_ = mymodule(imgs)
            test_result_loss = loss(outputs_, label)
            right_number += (outputs_.argmax(1) == label).sum()

        # writer.add_scalar("在测试集上的准确率",(right_number/test_len),(i+1))
        print("第{}轮训练在测试集上的准确率为{}".format((i + 1), (right_number / test_len)))"""

"""
https://blog.csdn.net/FUTEROX/article/details/122724634
"""
