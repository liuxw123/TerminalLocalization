import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np


from test20191119.dataset import PstData
from test20191119.modelDef import PstModel, PstLoss

lr = 0.0001
batch = 16
model = PstModel()

# model.to(DEVICE)

dataSet = PstData("train")

optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.0001)
loader = DataLoader(dataSet, batch_size=batch, shuffle=True, num_workers=0, collate_fn=collate)

lossFun = PstLoss()


def lrAdjust(epoch):

    if ((epoch+1) % 50000) == 0:
        num = (epoch + 1) // 50000
        newLr = lr / (np.power(10, num))

        for para in optimizer.param_groups:
            para['lr'] = newLr


def saveModel(epoch, precision):
    fileName = "models/pstModel{}_{:.2f}.pth".format(epoch, precision * 100)

    torch.save(model.state_dict(), fileName)




def train(numEpoch):
    for epoch in range(numEpoch):
        model.train()

        trainAcc = 0
        trainLoss = 0
        cnt = 0

        for data in loader:
            # print("*"*100)
            # print("*"*100)
            x, y = data
            optimizer.zero_grad()
            out = model(x)

            loss = lossFun(out, y)
            loss.backward()

            optimizer.step()
            #
            # for para in model.parameters():
            #     pass

            # print(loss)

            trainLoss += loss.item()
            cnt += 1

            print("loss:{:.4f}".format(loss.item()))

        lrAdjust(epoch)
        print("epoch: {:>5d},loss: {:.4f}".format(epoch, trainLoss / cnt))



if __name__ == '__main__':
    train(200000)

