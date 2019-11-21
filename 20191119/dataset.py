from torch.utils.data import Dataset

from DataOprt.preProcess import CsiDataPreProcess
from config import *

import numpy as np


class PstData(Dataset):

    def __init__(self, phase: str):

        assert phase in ["train", "test"]

        self.phase = phase

        self.origData = np.ndarray((numTestPoint, numAntenna, (1 + antennaDataIsCplx)))
        self.eny = np.ndarray(numTestPoint)
        self.pst = np.ndarray((numTestPoint, 2))

        self.train = CsiDataPreProcess.randomChoose("train")
        self.test = CsiDataPreProcess.randomChoose("test")

        for i in range(numTestPoint):
            fileName = DATA_DIR + "{}.txt".format(i)

            data = CsiDataPreProcess.readBestCsiData(fileName)
            self.origData[data["serNum"]] = data["csi"]
            self.eny[data["serNum"]] = data["eny"]
            self.pst[data["serNum"]] = data["pst"]

        self.corr = CsiDataPreProcess.getMatrixCorrelation(self.origData)

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

        nums = self.train[index]

        tempCorr = self.corr[nums[-1]]
        corrData = tempCorr[nums[:-1]]
        enyData = self.eny[nums]

        xData = np.hstack((corrData, enyData))

        lengths = self.computeLength(nums)

        nSum = lengths.sum()

        temp = []
        for i in range(numKnownPoint):
            temp.append(nSum / lengths[i])

        lengths = np.asarray(temp)
        nSum = lengths.sum()

        for i in range(numKnownPoint):
            lengths[i] = lengths[i] / nSum

        return {"x": xData, "y": lengths}

    def __len__(self) -> int:
        return self.train.shape[0] if self.phase == "train" else self.test.shape[0]


def collate(batch):
    pass

data = PstData("train")
aa = data[0]

