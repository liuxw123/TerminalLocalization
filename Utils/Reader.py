from Utils.FileOprt import FileOprter
# from Utils.DataOprt import DataOprt

import numpy as np
from numpy import random

from config import *


class DataReader:

    def __init__(self, antennaNum=8, userNum=20, testPointNum=264, times=10, selectionNum=20) -> None:
        super().__init__()

        self.values = {}
        self.antennaNum = antennaNum
        self.userNum = userNum
        self.testPointNum = testPointNum
        self.times = times

        for i in range(self.testPointNum):
            key = "p" + str(i + 1)
            self.values[key] = None

        # 选20个已知点
        self.selectionNum = selectionNum

    @staticmethod
    def __read(file, mode="CSI"):

        fr = open(file, "r")
        contents = fr.readlines()
        fr.close()

        ans = []
        for item in contents:
            if mode == "CSI":
                ans.append(float(item))
            else:
                temp = item.split(" ")
                if len(temp) == 2:
                    ans.append(float(temp[0]))
                    ans.append(float(temp[1]))
                else:
                    print("文件：" + file + " 有非法行")

        return np.asarray(ans)

    def chooseMaxEnergy(self, data):

        length = data.shape[0]
        dataR = data[0:length:2]  # 160个数 实部
        dataI = data[1:length:2]  # 160个数 虚部

        dataC = dataR + 1j * dataI  # 160 个复数
        eny = np.ndarray(self.userNum)

        for i in range(self.userNum):
            temp = dataC[i * self.antennaNum:i * self.antennaNum + self.antennaNum]
            eny[i] = temp.dot(temp.conj()).real

        idx = eny.argmax()

        return dataC[idx * self.antennaNum:idx * self.antennaNum + self.antennaNum], idx

    @staticmethod
    def correlation(data):
        n, _ = data.shape

        res = np.ndarray((n, n))

        for i in range(n):
            a1 = data[i]
            for j in range(n):
                a2 = data[j]
                res[i, j] = np.abs(a1.dot(a2.conj())) / np.sqrt((np.abs(a1.dot(a1.conj())))) / np.sqrt(
                    (np.abs(a2.dot(a2.conj()))))

        return res

    def chooseBestCorrelation(self, data):

        ans = np.ndarray((self.testPointNum, self.antennaNum), dtype=np.complex)
        ids = np.ndarray(self.testPointNum, dtype=np.int)

        for i in range(self.testPointNum):
            tempData = data[i]
            corr = self.correlation(tempData)
            corrSum = np.sum(corr, axis=1)

            idx = np.argmax(corrSum)
            bestCorr = corr[idx, :]
            # length[i], ans[:, i] = self.compute(tempData[:, idx])
            t = np.sum(bestCorr > 0.9)

            print(str(i + 1) + ":" + str(t) + "个")

            if t < 4:
                print(str(i + 1) + "号位置相关系数大于0.9:" + str(t) + "个")

            ans[i, :] = tempData[idx, :]
            ids[i] = idx

        return ans, ids

    def readCSIData(self, path):

        assert FileOprter.isDir(path) is True

        if not path.endswith("/"):
            path = path + "/"

        dataS = np.ndarray((self.testPointNum, self.times, self.antennaNum), dtype=np.complex)
        ids = np.ndarray((self.testPointNum, self.times), dtype=np.int)

        for i in range(self.testPointNum):
            arr = np.ndarray((self.times, self.antennaNum), dtype=np.complex)
            for j in range(self.times):
                file = path + str(i + 1) + "_" + str(j + 1) + ".txt"

                if not FileOprter.exist(file):
                    print("文件：" + file + " 不存在，终止！")
                    return

                arr[j, :], idx = self.chooseMaxEnergy(self.__read(file, mode="CSI"))
                ids[i, j] = idx

            dataS[i, :, :] = arr

        return self.chooseBestCorrelation(dataS), ids

    @staticmethod
    def readPositionData(path):
        fr = open(path)
        contents = fr.readlines()
        fr.close()

        positions = []
        for item in contents:
            xPos, yPos = item.split(" ")
            xPos = int(xPos)
            yPos = int(yPos)

            dot = [xPos, yPos]
            positions.append(dot)

        return positions

    def energy(self, data):

        absSum = 0

        for i in range(self.antennaNum):
            absSum += np.abs(data[i])

        return absSum

    def package(self, csi, positions, ids1, ids2):

        for i in range(self.testPointNum):
            info = "File:{0}_{1}.txt\nLine:{2} -- {3}".format(str(i + 1), str(ids2[i] + 1),
                                                              str((ids1[i, ids2[i]]) * 16 + 1),
                                                              str((ids1[i, ids2[i]]) * 16 + 16))
            key = "p" + str(i + 1)
            val = {"position": positions[i], "CSI": csi[i], "energy": self.energy(csi[i]),
                   "file": "{0}_{1}.txt".format(i + 1, ids2[i] + 1),
                   "line": (ids1[i, ids2[i]]) * 16 + 1}

            self.values[key] = val

    @staticmethod
    def writeMaxEnergyData(ids):
        m, n = ids.shape

        for i in range(m):
            for j in range(n):
                fileName = CSIDIR + str(i+1) + "_" + str(j+1) + ".txt"

                fr = open(fileName, "r")
                lines = fr.readlines()
                fr.close()





    # 获取原始数据信息
    def get(self, CSIDir, positionFile):

        data, ids2 = self.readCSIData(CSIDir)
        self.writeMaxEnergyData(ids2)

        positions = self.readPositionData(positionFile)

        self.package(data[0], positions, ids2, data[1])

        return self.values

    @staticmethod
    def changeComplexToMatrix(data):

        ans = np.ndarray((2, data.shape[0]), dtype=np.float)

        real = data.real
        imag = data.imag

        ans[0] = real
        ans[1] = imag

        return ans

    def getCSIData(self, nums, target):

        n = len(nums) + 1

        targetKey = "p" + str(target)
        targetCSI = self.values[targetKey]["CSI"]  # 8 complex

        targetCSI = self.changeComplexToMatrix(targetCSI)

        ans = np.ndarray((n, 2, self.antennaNum))
        ans[-1] = targetCSI

        for i, num in enumerate(nums):
            pointKey = "p" + str(num)
            pointCSI = self.values[pointKey]["CSI"]
            pointCSI = self.changeComplexToMatrix(pointCSI)

            ans[i] = pointCSI

        return ans

    def computeLength(self, nums, target):

        targetKey = "p" + str(target)
        targetPos = self.values[targetKey]["position"]
        x1 = targetPos[0]
        y1 = targetPos[1]

        lengths = []

        for num in nums:
            pointKey = "p" + str(num)
            pointPos = self.values[pointKey]["position"]

            x2 = pointPos[0]
            y2 = pointPos[1]

            lengths.append((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

        return np.asarray(lengths)

    # 随机选择20个点和target
    def choose(self):

        nums = []

        idx = 0

        while idx < self.selectionNum:

            while True:
                num = random.randint(self.testPointNum)

                try:
                    nums.index(num)
                except ValueError:
                    nums.append(num)
                    break

            idx += 1

        nums.sort()
        nums.append(random.randint(self.testPointNum))

        return nums

    # 为断点继续训练使用
    @staticmethod
    def durability(data):

        file = "/home/lxw/PycharmProjects/TerminalLocalization/datas/nums.txt"

        if FileOprter.exist(file):
            ch = input("持久化文件已经存在，是否将其覆盖，这可能会导致上次训练过程丢失。[Y/n]")

            if ch.lower() == "n" or ch.lower() == "no":
                print("已返回，持久化文件未覆盖")
                return

        string = ""
        for item in data:

            for num in item:
                string = string + str(num) + " "

            string = string[:-1] + "\n"

        fw = open(file, "w+")
        fw.writelines(string[:-1])
        fw.close()

    def generate(self, total):

        idx = 0
        nums = []

        while idx < total:

            while True:
                try:
                    item = self.choose()
                    nums.index(item)
                except ValueError:
                    nums.append(item)
                    break

            idx += 1

        self.durability(nums)


def printToFile(data):
    for key in data.keys():

        string = ""

        item = data[key]
        fileName = "/home/lxw/PycharmProjects/TerminalLocalization/datas/format/" + key + ".txt"
        csi = item["CSI"]
        pst = item["position"]
        eny = item["energy"]
        fileFrom = item["file"]
        lineFrom = item["line"]

        string += "Serial Number: " + key[1:] + "\n"
        string += "Position: ({0}, {1})".format(pst[0], pst[1]) + "\n"
        string += "Choose From: " + fileFrom + " / {0} -- {1}".format(lineFrom, lineFrom + 15) + "\n"
        string += "Energy: {:.4f}".format(eny) + "\n"
        string += "=" * 50 + "\n"
        string += "CSI Data Information:" + "\n"
        string += "=" * 50 + "\n"

        real = csi.real
        imag = csi.imag
        n = real.shape[0]

        for j in range(n):
            realStr = "{}".format(real[j])
            imagStr = "{}".format(imag[j])
            string += " "*3 + " "*(12-len(realStr)) + realStr + " "*3 + imagStr + "\n"


        string += "=" * 50 + "\n"

        fw = open(fileName, "w+")
        fw.writelines(string)
        fw.close()





dir1 = "/home/lxw/PycharmProjects/TerminalLocalization/datas/fortest/"
file = "/home/lxw/PycharmProjects/TerminalLocalization/datas/position.txt"

dr = DataReader()
ans = dr.get(dir1, file)
#
# print(dr.computeLength(dr.choose()))

printToFile(ans)
