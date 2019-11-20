from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QApplication, QMainWindow, QAbstractItemView

from config import *

import numpy as np

from Ui.watchData import Ui_MainWindow

from DataOprt.preProcess import CsiDataPreProcess


class Window(QMainWindow):

    def __init__(self) -> None:
        super().__init__()

        self.mainWin = Ui_MainWindow()
        self.mainWin.setupUi(self)
        self.initGui()

        self.mainWin.showTableView.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # self.initOrigData()
        self.signalConnect()

        self.psts = CsiDataPreProcess.readPstFile()

    def signalConnect(self):

        self.mainWin.indexComboBox.currentIndexChanged.connect(self.slotForComboBox)
        self.mainWin.chgLineEdit.returnPressed.connect(self.jumpIndexToShow)
        self.mainWin.saveLineEdit.returnPressed.connect(self.saveLineEdit)

    def saveLineEdit(self):
        text = self.mainWin.saveLineEdit.text()

        try:
            idx = int(text)
            assert testTimes > idx >= 0
            self.save(idx)

            self.mainWin.saveLineEdit.setText("")
            self.changePointToShow(self.mainWin.indexComboBox.currentIndex()+1)
        except Exception:
            self.mainWin.saveLineEdit.setText("不符合")

    def jumpIndexToShow(self):

        text = self.mainWin.chgLineEdit.text()

        try:
            idx = int(text)
            assert numTestPoint > idx >= 0
            self.changePointToShow(idx + 1)
            self.mainWin.chgLineEdit.setText("")
        except Exception:
            self.mainWin.chgLineEdit.setText("不符合")

    def slotForComboBox(self, num):
        # print(num)
        self.changePointToShow(num)

    def initGui(self):

        model = QStandardItemModel()

        for i in range(testTimes):
            model.setHorizontalHeaderItem(i, QStandardItem(str(i)))
            model.setVerticalHeaderItem(i, QStandardItem(str(i)))

        model.setVerticalHeaderItem(testTimes, QStandardItem("Eny"))
        model.setVerticalHeaderItem(testTimes + 1, QStandardItem("排名"))
        model.setVerticalHeaderItem(testTimes + 2, QStandardItem("Sum"))
        model.setVerticalHeaderItem(testTimes + 3, QStandardItem("排名"))

        self.mainWin.showTableView.setModel(model)

        self.mainWin.indexComboBox.addItem("请选择")
        for i in range(numTestPoint):
            self.mainWin.indexComboBox.addItem(str(i))

    @staticmethod
    def readOnePointData(num: int) -> {}:

        csi = np.ndarray((testTimes, numAntenna, (1 + antennaDataIsCplx)))
        eny = np.ndarray(testTimes)

        start = num * testTimes + 1
        end = (num + 1) * testTimes + 1

        for i in range(start, end):
            fileName = CHS_DIR + "{}.txt".format(i)
            res = CsiDataPreProcess.readMaxEnergyCsiFile(fileName)
            csi[i - start] = res["csi"]
            eny[i - start] = res["eny"]

        return {"csi": csi, "eny": eny}

    def changePointToShow(self, currPointNum):
        # currPointNum = self.mainWin.indexComboBox.currentIndex()

        if currPointNum < 1 or currPointNum > numTestPoint:
            return

        print("DEBUG:"+str(currPointNum))
        self.mainWin.indexComboBox.setCurrentIndex(currPointNum)

        xPos = self.psts[currPointNum-1][0]
        yPos = self.psts[currPointNum-1][1]

        self.mainWin.pstLabel.setText("({} , {})".format(xPos, yPos))

        res = self.readOnePointData(currPointNum - 1)
        eny = res["eny"]
        ids = np.argsort(eny)
        ans = CsiDataPreProcess.getOnePointCorrelation(res["csi"])
        idx = np.argsort(-ans.sum(axis=1))

        model = self.mainWin.showTableView.model()

        for i in range(testTimes):
            for j in range(testTimes):
                model.setItem(i, j, QStandardItem("{:.6f}".format(ans[i, j])))

            model.setItem(testTimes, i, QStandardItem("{:.2f}".format(eny[i])))
            model.setItem(testTimes + 1, ids[i], QStandardItem("{}".format(testTimes - i)))

            model.setItem(testTimes + 2, i, QStandardItem("{:.6f}".format(ans[i].sum())))
            model.setItem(testTimes + 3, idx[i], QStandardItem("{}".format(i + 1)))

    def save(self, idx):

        fileIndex = self.mainWin.indexComboBox.currentIndex() - 1

        fileName = CHS_DIR + "{}.txt".format(fileIndex * testTimes + idx + 1)
        # print(fileName)

        ans = CsiDataPreProcess.readMaxEnergyCsiFile(fileName)

        CsiDataPreProcess.writeBestCsiData(ans["csi"], self.psts[fileIndex], fileIndex, idx, ans["eny"])

    # def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
    #
    #     print(a0.key())
    #     print(a0.key() == Qt.Key_L)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    app.setApplicationName("watchData")

    win = Window()
    win.show()
    sys.exit(app.exec_())
