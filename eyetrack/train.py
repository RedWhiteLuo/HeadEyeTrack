import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

trans = transforms.Compose([transforms.ToTensor()])
# 获取训练集
dataset = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=trans, download=True)
dataset2 = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=trans, download=True)
train_dataloader = DataLoader(dataset, batch_size=64)
test_dataloader = DataLoader(dataset2, batch_size=64)

test_len = len(dataset2)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("使用GPU训练中：{}".format(torch.cuda.get_device_name()))
else:
    device = torch.device("cpu")
    print("使用CPU训练")


class MyModule(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = Sequential(
            Conv2d(3, 32, kernel_size=(5, 5), padding=2),   # keep W H
            MaxPool2d(2),
            Conv2d(32, 32, (5, 5), padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, (5, 5), padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)

        return x


if __name__ == '__main__':
    writer = SummaryWriter()
    mymodule = MyModule()
    mymodule = mymodule.to(device)  # 模型转移GPU
    loss = torch.nn.CrossEntropyLoss()
    learn_step = 0.01
    optim = torch.optim.SGD(mymodule.parameters(), lr=learn_step)
    epoch = 1000

    train_step = 0  # 每轮训练的次数
    mymodule.train()  # 模型在训练状态
    for i in range(epoch):
        print("第{}轮训练".format(i + 1))
        train_step = 0
        for data in train_dataloader:
            imgs, targets = data

            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = mymodule(imgs)
            result_loss = loss(outputs, targets)
            optim.zero_grad()
            result_loss.backward()
            optim.step()

            train_step += 1
            if train_step % 100 == 0:
                print("第{}轮的第{}次训练的loss:{}".format((i + 1), train_step, result_loss.item()))

        # 在测试集上面的效果
        mymodule.eval()  # 在验证状态
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
            print("第{}轮训练在测试集上的准确率为{}".format((i + 1), (right_number / test_len)))

        if (i + 1) % 10 == 0:
            # 保存模型
            torch.save(mymodule, "mymodule_{}.pth".format((i + 1)))
            print("a")



"""
https://blog.csdn.net/FUTEROX/article/details/122724634
"""