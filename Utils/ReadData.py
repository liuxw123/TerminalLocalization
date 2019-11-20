import os.path as osp
import numpy as np

from Utils.DataOprt import DataOprt


class ReadData:
    FILE_SPLITOR = ["enter", "blank"]
    N_ANTENNA = 8  # 天线数
    N_USER = 20  # AUserDetection输出数
    N_TEST_POINT = 2640  # 测试次数
    N_TEST_POINT_FOR_ONE = 10  # 一个点的采集次数

    def checkFileValid(self, file):
        flag = True
        try:
            assert type(file) is str
            assert osp.exists(file)
        except TypeError:
            print("file is not str type!")
            flag = False
        except FileNotFoundError:
            print("file is not exist.file:" + file)
            flag = False

        return flag

    def readDecimalData(self, file, split="Enter"):

        try:
            assert file is not None
        except Exception as e:
            print("file is None.")
            return

        try:
            assert type(split) is str

            if split.lower() not in ReadData.FILE_SPLITOR:
                raise Exception()
        except Exception:
            print("splitor is unsupported type: " + split)
            return

        split = split.lower()

        if split == "enter":
            fr = open(file, "r")
            datas = fr.readlines()
            fr.close()
        elif split == "blank":
            pass

        ans = []
        for item in datas:
            ans.append(float(item))

        return np.asarray(ans)

    def constructComplex(self, data):
        try:
            assert data is not None
            assert type(data) is np.ndarray
        except Exception:
            print("input data is None or not a array.")

        len = data.shape[0]
        datar = data[0:len:2]  # 160个数 实部
        datai = data[1:len:2]  # 160个数 虚部

        datac = datar + 1j * datai  # 160 个复数
        eny = np.ndarray(ReadData.N_USER)

        for i in range(0, 20):
            temp = datac[i * ReadData.N_ANTENNA:i * ReadData.N_ANTENNA + ReadData.N_ANTENNA]
            eny[i] = temp.dot(temp.conj()).real

        idx = eny.argmax()
        return datac[idx * ReadData.N_ANTENNA:ReadData.N_ANTENNA + idx * ReadData.N_ANTENNA]

    def getData(self, filePath):
        arr = np.ndarray((ReadData.N_ANTENNA, ReadData.N_TEST_POINT), dtype=np.complex)

        for i in range(ReadData.N_TEST_POINT):
            file = filePath + str(i + 1) + ".txt"
            ans = self.readDecimalData(file)
            data = self.constructComplex(ans)
            arr[:, i] = data

        return self.chooseBest(arr)

    def compute(self, data):

        absSum = 0

        for i in range(ReadData.N_ANTENNA):
            absSum += np.abs(data[i])

        for i in range(ReadData.N_ANTENNA):
            data[i] = data[i] / absSum

        return 20 * np.log10(absSum), data

    def chooseBest(self, data):

        n = ReadData.N_TEST_POINT // ReadData.N_TEST_POINT_FOR_ONE
        ans = np.ndarray((ReadData.N_ANTENNA, n), dtype=np.complex)
        length = np.ndarray(n)
        for i in range(n):
            tempData = data[:, i * ReadData.N_TEST_POINT_FOR_ONE:(i + 1) * ReadData.N_TEST_POINT_FOR_ONE]

            # ReadData.printMatrix(DataOprt.correlation(tempData), numPos=3)
            corr = DataOprt.correlation(tempData)
            corrSum = np.sum(corr, axis=1)

            # ReadData.printArr(corrSum, numPos=3)
            # ReadData.printMatrix(corr, numPos=3)
            idx = np.argmax(corrSum)
            bestCorr = corr[idx, :]
            length[i], ans[:, i] = self.compute(tempData[:, idx])
            t = np.sum(bestCorr > 0.9)

            print(str(i + 1) + ":" + str(t) + "个")

            if t < 4:
                print(str(i + 1) + "号位置相关系数大于0.9:" + str(t) + "个")
        return {"absSum": length, "chn": ans}

    @staticmethod
    def printArr(data, numPos=2, title="None"):
        numPos = "{:." + str(numPos) + "f}"
        n = data.shape[0]

        start = "| index |"
        string = "|   0   |"

        numStrList = []
        maxLength = 0
        for i in range(n):
            numStrList.append(numPos.format(data[i]))
            if len(numStrList[-1]) > maxLength:
                maxLength = len(numStrList[-1])

        maxLength += 3

        for i in range(n):
            idx = str(i)
            blank = " " * ((maxLength - len(idx)) // 2)

            if len(blank + idx + blank) + 1 == maxLength:
                start += blank + idx + blank + "|"
            else:
                start += blank + idx + blank[0:-1] + "|"

            blank = " " * ((maxLength - len(numStrList[i])) // 2)

            if len(blank + numStrList[i] + blank) + 1 == maxLength:
                string += blank + numStrList[i] + blank + "|"
            else:
                string += blank + numStrList[i] + blank[0:-1] + "|"

        string = "Title:" + title + "\n" + "=" * len(start) + "\n" + start + "\n" + "=" * len(
            start) + "\n" + string + "\n" + "=" * len(start) + "\n"
        print(string)

    @staticmethod
    def printMatrix(data, numPos=2, title="None"):
        numPos = "{:." + str(numPos) + "f}"
        m, n = data.shape

        numStrList = []
        maxLength = 0
        for i in range(m):
            tempList = []
            for j in range(n):
                tempList.append(numPos.format(data[i, j]))
                if len(tempList[-1]) > maxLength:
                    maxLength = len(tempList[-1])
            numStrList.append(tempList)

        maxLength += 3

        start = "| index |"
        num = len(start) - 1
        string = ""

        for i in range(n):
            idx = str(i)
            blank = " " * ((maxLength - len(idx)) // 2)

            if len(blank + idx + blank) + 1 == maxLength:
                start += blank + idx + blank + "|"
            else:
                start += blank + idx + blank[0:-1] + "|"

        for i in range(m):
            tempList = numStrList[i]
            idx = str(i)
            blank = " " * ((num - len(idx)) // 2)

            if len(blank + idx + blank) + 1 == num:
                string += "|" + blank + idx + blank + "|"
            else:
                string += "|" + blank + idx + blank[0:-1] + "|"
            for j in range(n):
                blank = " " * ((maxLength - len(tempList[j])) // 2)

                if len(blank + tempList[j] + blank) + 1 == maxLength:
                    string += blank + tempList[j] + blank + "|"
                else:
                    string += blank + tempList[j] + blank[0:-1] + "|"

            if i < m - 1:
                string += "\n" + '-' * len(start) + "\n"

        print("Title:" + title + "\n" + "=" * len(start) + "\n" + start + "\n" + "=" * len(
            start) + "\n" + string + "\n" + "=" * len(start) + "\n")
