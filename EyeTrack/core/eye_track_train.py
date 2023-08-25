import torch
from torch.utils.tensorboard import SummaryWriter
from EyeTrack.core.eye_track_model import EyeTrackModel
from Tools.dataloader import EpochDataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def eye_track_train():
    train_img, train_coords = EpochDataLoader(TRAIN_DATASET_PATH, batch_size=TRAIN_BATCH_SIZE)

    batch_num = train_img.size()[0]
    model = EyeTrackModel().to(device).train()
    loss = torch.nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=LEARN_STEP)
    writer = SummaryWriter(LOG_SAVE_PATH)

    trained_batch_num = 0
    for epoch in range(TRAIN_EPOCH):
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
            print("[INFO]: trained epoch num | trained batch num | loss "
                  , epoch + 1, trained_batch_num, result_loss.item())
        # if epoch // 50 == 0:
        #     torch.save(model, "../model/ET-" + str(epoch) + ".pt")
    # save model
    torch.save(model, "../model/ET-last.pt")
    writer.close()
    print("[SUCCEED!] model saved!")


if __name__ == "__main__":
    TRAIN_DATASET_PATH = 'D:/0_AI_Learning/HeadEyeTrack/dataset'
    LOG_SAVE_PATH = '../train_logs'
    TRAIN_BATCH_SIZE = 256
    LEARN_STEP = 0.01
    TRAIN_EPOCH = 1500
    eye_track_train()
