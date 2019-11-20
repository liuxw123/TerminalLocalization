from config import *

import numpy as np
from numpy import random


# TODO explain this file to do


class CsiDataPreProcess:

    # 读取文件，仅仅去除回车符，返回出list
    @staticmethod
    def __readFile(filePath: str) -> list:
        fr = open(filePath, "r")
        lines = fr.readlines()
        fr.close()

        ans = []
        for line in lines:
            ans.append(line.strip())

        return ans

    # 读取csi txt文件，返回np array数组， 20 * 8 * 2
    @staticmethod
    def __readCsiFile(filePath: str) -> np.ndarray:

        lines = CsiDataPreProcess.__readFile(filePath)

        assert len(lines) == numAntenna * (1 + antennaDataIsCplx) * numUserOutput  # 20 * 8 * 2
        ans = np.ndarray((numUserOutput, numAntenna, (1 + antennaDataIsCplx)))  # 20 * 8 * 2

        for idx, item in enumerate(lines):
            num = float(item)
            i = idx // (numAntenna * (1 + antennaDataIsCplx))
            j = (idx - i * (numAntenna * (1 + antennaDataIsCplx))) // (1 + antennaDataIsCplx)
            k = (idx - i * (numAntenna * (1 + antennaDataIsCplx))) % (1 + antennaDataIsCplx)
            ans[i, j, k] = num

        return ans

    # 计算序列能量 仅支持 8 * 2的向量序列
    @staticmethod
    def computeEnergy(data: np.ndarray) -> float:
        m, n = data.shape  # 支持二维 8 * 2

        assert m == numAntenna and n == (1 + antennaDataIsCplx)

        temp = data[:, 0] + 1j * data[:, 1]
        return np.sqrt(temp.dot(temp.conj()).real)

    # 获取一个csi data中最大能量的那个序号 20 * 8 * 2
    @staticmethod
    def __getMaxEnergy(data: np.ndarray):
        m, n, p = data.shape

        assert m == numUserOutput and n == numAntenna and p == (1 + antennaDataIsCplx)

        maxE = 0
        maxI = -1

        for i in range(m):

            eny = CsiDataPreProcess.computeEnergy(data[i])

            if eny > maxE:
                maxE = eny
                maxI = i

        return maxI, maxE

    # 将最大能量的那个数据，写入文件中
    @staticmethod
    def __writeMaxEnergyCsiFile(data: np.ndarray, idx1: int, idx2: int, best: int, eny: float):
        m, n = data.shape  # 支持二维 8 * 2

        assert m == numAntenna and n == (1 + antennaDataIsCplx)

        fileName = CHS_DIR + "{}".format(idx1 * testTimes + idx2 + 1) + ".txt"

        string = "Serial Number: {} -- {}\n".format(idx1, idx2)
        string += "Choose From: {}.txt / {} -- {}\n".format(idx1 * testTimes + idx2 + 1,
                                                            best * numAntenna * (1 + antennaDataIsCplx) + 1,
                                                            (best + 1) * numAntenna * (1 + antennaDataIsCplx))
        string += "Energy: {}\n".format(eny)
        string += "=" * 50 + "\n"
        string += "CSI Data Information:\n"
        string += "=" * 50 + "\n"

        for i in range(m):
            realStr = "{}".format(data[i, 0])
            imagStr = "{}".format(data[i, 1])
            string += " " * 3 + " " * (12 - len(realStr)) + realStr + " " * 3 + imagStr + "\n"

        string += "=" * 50 + "\n"

        fw = open(fileName, "w+")
        fw.writelines(string)
        fw.close()

    # 将最优的csi data写入到format/文件夹中
    @staticmethod
    def writeBestCsiData(csi: np.ndarray, pst: list, idx: int, chs: int, eny: float):
        m, n = csi.shape

        assert m == numAntenna and n == (1 + antennaDataIsCplx)
        fileName = DATA_DIR + "{}".format(idx) + ".txt"

        string = "Serial Number: {}\n".format(idx)

        string += "Position: ({},{})\n".format(pst[0], pst[1])
        string += "Choose From: {}{}.txt\n".format(CHS_DIR, idx * testTimes + chs + 1)

        string += "Energy: {}\n".format(eny)
        string += "=" * 50 + "\n"
        string += "CSI Data Information:\n"
        string += "=" * 50 + "\n"

        for i in range(m):
            realStr = "{}".format(csi[i, 0])
            imagStr = "{}".format(csi[i, 1])
            string += " " * 3 + " " * (12 - len(realStr)) + realStr + " " * 3 + imagStr + "\n"

        string += "=" * 50 + "\n"

        fw = open(fileName, "w+")
        fw.writelines(string)
        fw.close()

    # 将format/文件夹中最优的csi data读出
    @staticmethod
    def readBestCsiData(filePath):

        lines = CsiDataPreProcess.__readFile(filePath)

        serNum = lines[0].split(": ")[-1]
        serNum = int(serNum)

        pst = lines[1].split(": ")[-1]
        pst = pst[1:-1].split(",")
        pst = [int(x) for x in pst]

        contents = lines[7:7 + numAntenna]
        csi = np.ndarray((numAntenna, (1 + antennaDataIsCplx)))

        for i, item in enumerate(contents):

            temp = item.strip(" ")
            temp = temp.split(" ")

            if antennaDataIsCplx:

                real = float(temp[0])
                imag = float(temp[-1])
                csi[i, 0] = real
                csi[i, 1] = imag
            else:

                real = float(temp[0])

                csi[i, 0] = real

        eny = CsiDataPreProcess.computeEnergy(csi)

        ans = {"csi": csi, "eny": eny, "serNum": serNum, "pst": pst}
        return ans

    # 从 choose/文件夹读取csi数据
    @staticmethod
    def readMaxEnergyCsiFile(filePath: str) -> {}:

        lines = CsiDataPreProcess.__readFile(filePath)

        contents = lines[6:6 + numAntenna]

        csi = np.ndarray((numAntenna, (1 + antennaDataIsCplx)))

        for i, item in enumerate(contents):

            temp = item.strip(" ")
            temp = temp.split(" ")

            if antennaDataIsCplx:

                real = float(temp[0])
                imag = float(temp[-1])
                csi[i, 0] = real
                csi[i, 1] = imag
            else:

                real = float(temp[0])

                csi[i, 0] = real

        serialNum = lines[0]
        eny = CsiDataPreProcess.computeEnergy(csi)

        serialNum = serialNum.split(" ")

        ans = {"csi": csi, "eny": eny, "serNum": serialNum[-3] + "_" + serialNum[-1]}

        return ans

    # 从UserDetection中选取最大能量，写入choose/文件夹中，写入后，并读出对比，验证写入正确
    @staticmethod
    def chooseMaxEnergy():

        for i in range(numTestPoint):
            for j in range(testTimes):
                fileName = CSI_DIR + "{}.txt".format(i * testTimes + j + 1)

                data = CsiDataPreProcess.__readCsiFile(fileName)
                idx, eny = CsiDataPreProcess.__getMaxEnergy(data)

                CsiDataPreProcess.__writeMaxEnergyCsiFile(data[idx], i, j, idx, eny)

                fileName1 = CHS_DIR + "{}.txt".format(i * testTimes + j + 1)

                data1 = CsiDataPreProcess.readMaxEnergyCsiFile(fileName1)

                res = np.abs(data[idx] - data1["csi"]).sum()

                if res > 0:
                    print("Error! file:" + fileName)

    @staticmethod
    def correlation(data1: np.ndarray, data2: np.ndarray) -> float:
        m, n = data1.shape
        assert m == numAntenna and n == (1 + antennaDataIsCplx)

        m, n = data2.shape
        assert m == numAntenna and n == (1 + antennaDataIsCplx)

        if antennaDataIsCplx:
            temp1 = data1[:, 0] + 1j * data1[:, 1]
            temp2 = data2[:, 0] + 1j * data2[:, 1]

            res = np.abs(temp1.dot(temp2.conj())) / np.sqrt((np.abs(temp1.dot(temp1.conj())))) / np.sqrt(
                (np.abs(temp2.dot(temp2.conj()))))
        else:
            res = 1

        return res

    # 获取矩阵间向量的相关系数矩阵
    @staticmethod
    def getMatrixCorrelation(data: np.ndarray) -> np.ndarray:
        m, n, p = data.shape

        assert n == numAntenna and p == (1 + antennaDataIsCplx)

        ans = np.ndarray((m, m))
        for i in range(m):

            a1 = data[i]
            for j in range(m):
                a2 = data[j]
                ans[i, j] = CsiDataPreProcess.correlation(a1, a2)

        return ans

    # 获取在一个测试点上测试10次相关系数矩阵
    @staticmethod
    def getOnePointCorrelation(data1: np.ndarray) -> np.ndarray:

        m, n, p = data1.shape
        assert m == testTimes and n == numAntenna and p == (1 + antennaDataIsCplx)

        ans = np.ndarray((testTimes, testTimes))

        for i in range(testTimes):
            a1 = data1[i]

            for j in range(testTimes):
                a2 = data1[j]
                ans[i, j] = CsiDataPreProcess.correlation(a1, a2)

        return ans

    @staticmethod
    def readPstFile() -> np.ndarray:
        lines = CsiDataPreProcess.__readFile(PST_FILE)

        assert len(lines) == numTestPoint
        psts = np.ndarray((numTestPoint, 2), dtype=np.int)

        for i, item in enumerate(lines):
            temp = item.strip(" ").split(" ")
            x = int(temp[0])
            y = int(temp[1])
            psts[i] = np.asarray([x, y])

        return psts

    # 随机选取点编号
    @staticmethod
    def randomChoose(phase: str):

        def choose():
            nums = []

            N = numKnownPoint
            while N > 0:
                number = random.randint(numTestPoint)

                try:
                    nums.index(number)
                except Exception:
                    nums.append(number)
                    N -= 1
            nums.sort()

            while True:
                number = random.randint(numTestPoint)

                try:
                    nums.index(number)
                except Exception:
                    nums.append(number)
                    break
            return nums

        ans = []
        if phase == "train":
            n = numTrainSample
        elif phase == "test":
            n = numTestSample

        while n > 0:
            num = choose()

            try:
                ans.index(num)
            except Exception:
                ans.append(num)
                n -= 1
        return np.asarray(ans)



# CsiDataPreProcess.chooseMaxEnergy()
