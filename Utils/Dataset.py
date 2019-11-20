# from torch.utils.data.dataset import T_co

import config

# from Utils.Reader import DataReader
from Utils.FileOprt import FileOprter

from DataOprt.preProcess import CsiDataPreProcess
from config import *

import torch
from torch.utils.data import Dataset
# from torch.utils.data.dataset import T_co

# import random
import numpy as np
from numpy import random


class DataReader:

    def __init__(self, path) -> None:
        super().__init__()

        self.total = config.numTestPoint
        self.data = {}
        self.read(path)

    @staticmethod
    def format(string):

        assert type(string) is str

        temp = string.replace("  ", " ")

        while temp != string:
            string = temp
            temp = string.replace("  ", " ")

        return string[1:]

    @staticmethod
    def readPst(string):

        pos = string[string.index("(") + 1:string.index(")")].split(", ")

        return int(pos[0]), int(pos[1])



    def read(self, path):

        for i in range(self.total):
            file = path + "p" + str(i+1) + ".txt"
            fr = open(file)
            contents = fr.readlines()
            fr.close()

            strings = contents[7:15]
            real = []
            imag = []
            for item in strings:

                string = self.format(item).split(" ")

                real.append(float(string[0]))
                imag.append(float(string[1]))



            key = "p" + contents[0][contents[0].index(":")+2:-1]
            val = {"csi": np.asarray([real, imag]), "pst": self.readPst(contents[1])}
            self.data[key] = val

    def getTorchData(self, nums: list):


        if nums[-1] in nums[:-1]:
            idx = nums[:-1].index(nums[-1])
            yLabel = np.zeros((len(nums[:-1])), dtype=np.float32)
            yLabel[idx] = 1
        else:
            lengths = torch.tensor(self.computeLength(nums[:-1], nums[-1] + 1), dtype=torch.float32)

            absSum = lengths.dot(lengths)
            yLabel = []

            for i in range(lengths.shape[0]):
                yLabel.append(absSum / (lengths[i] * lengths[i]))

            nSum = sum(yLabel)
            temp = []

            for num in yLabel:
                temp.append(num / nSum)

            yLabel = temp





        csiData = []

        for num in nums:
            key = "p" + str(num+1)
            item = self.data[key]
            csi = item["csi"]
            csiData.append(csi)

        csiData = np.asarray(csiData)

        xData = torch.tensor(csiData, dtype=torch.float32)

        return xData.permute(1, 0, 2), torch.tensor(np.asarray(yLabel), dtype=torch.float32)


    def computeLength(self, nums, target):

        targetKey = "p" + str(target)
        targetPos = self.data[targetKey]["pst"]
        x1 = targetPos[0]
        y1 = targetPos[1]

        lengths = []

        for num in nums:
            pointKey = "p" + str(num+1)
            pointPos = self.data[pointKey]["pst"]

            x2 = pointPos[0]
            y2 = pointPos[1]

            lengths.append((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

        return np.asarray(lengths)


class PstData(Dataset):

    def __init__(self, phase) -> None:
        super().__init__()

        assert phase in ["train", "test"]

        # self.dr = DataReader(path)

        self.origData = np.ndarray((numTestPoint, numAntenna, (1+antennaDataIsCplx)))
        # self.eny = np.ndarray(numTestPoint)
        self.pst = np.ndarray((numTestPoint, 2))

        self.phase = phase

        self.train = CsiDataPreProcess.randomChoose("train")
        self.test = CsiDataPreProcess.randomChoose("test")

        for i in range(numTestPoint):
            fileName = DATA_DIR + "{}.txt".format(i)

            data = CsiDataPreProcess.readBestCsiData(fileName)
            self.origData[data["serNum"]] = data["csi"]
            # self.eny[data["serNum"]] = data["eny"]
            self.pst[data["serNum"]] = data["pst"]


    def changeMode(self, phase):
        assert phase in ["train", "test"]
        self.phase = phase

    def computeLength(self, nums: np.ndarray) -> np.ndarray:

        m = nums.shape[-1]
        assert m == (numKnownPoint + 1)
        target = nums[-1]
        target = self.pst[target]

        lengths = np.ndarray(numKnownPoint)

        for i in range(m-1):
            point = self.pst[nums[i]]
            lengths[i] = (target[0]-point[0])*(target[0]-point[0])+(target[1]-point[1])*(target[1]-point[1])

        return lengths



    def __getitem__(self, index: int):

        if self.phase == "train":
            nums = self.train[index]
        elif self.phase == "test":
            nums = self.test[index]

        # TODO 准备数据

        # print("{} ".format(index))

        xData = np.ndarray((numKnownPoint+numTarget, numAntenna, 1+antennaDataIsCplx))
        for i in range(numKnownPoint+numTarget):
            xData[i] = self.origData[nums[i]]

        tempNum1 = nums[:-1]
        tempNum1 = list(tempNum1)

        if nums[-1] in tempNum1:
            idx = tempNum1.index(nums[-1])
            yLabel = np.zeros(numKnownPoint, dtype=np.float)
            yLabel[idx] = 1
        else:
            lengths = self.computeLength(nums)

            nSum = lengths.sum()

            temp = []
            for i in range(numKnownPoint):
                temp.append(nSum / lengths[i])

            lengths = np.asarray(temp)
            nSum = lengths.sum()

            for i in range(numKnownPoint):
                lengths[i] = lengths[i] / nSum

        # xData, yLabel = self.dr.getTorchData(nums)

        return {"x": xData, "y": lengths}

    def __len__(self) -> int:

        if self.phase == "train":
            return self.train.shape[0]
        elif self.phase == "test":
            return self.test.shape[0]

        return 0


def collate(batch):

    nBatch = len(batch)
    d, w, h = batch[0]['x'].shape   # 21 8 2  --> 2 21 8

    xBatch = torch.rand((nBatch, d, w, h), dtype=torch.float32)
    yBatch = torch.rand((nBatch, batch[0]['y'].shape[0]), dtype=torch.float32)

    xBatch = np.ndarray((nBatch, d, w, h))
    yBatch = np.ndarray((nBatch, batch[0]['y'].shape[0]))

    for i, item in enumerate(batch):
        xBatch[i] = item['x']
        yBatch[i] = item['y']


    temp = xBatch
    xBatch = torch.tensor(xBatch, dtype=torch.float32).permute(0, 3, 1, 2)
    yBatch = torch.tensor(yBatch, dtype=torch.float32)

    temp1 = xBatch[:, 0, :, :]
    temp2 = xBatch[:, 1, :, :]

    xBatch1 = xBatch

    min1 = temp1.min()
    min2 = temp2.min()

    max1 = temp1.max()
    max2 = temp2.max()

    temp1 = (temp1 - min1) / (max1 - min1)
    temp2 = (temp2 - min2) / (max2 - min2)

    xBatch1[:, 0, :, :] = temp1
    xBatch1[:, 1, :, :] = temp2

    if torch.isnan(xBatch).sum() > 0:
        print("De")

    _, maxI = yBatch.max(dim=1)

    y = torch.zeros(yBatch.shape)

    for i in range(y.shape[0]):
        y[i, maxI[i]] = 1

    return xBatch1, y


