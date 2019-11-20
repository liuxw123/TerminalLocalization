from PyQt5.QtCore import QModelIndex
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QApplication, QMainWindow, QAbstractItemView

from Ui.checkData import Ui_MainWindow

from config import *

import numpy as np

from DataOprt.preProcess import CsiDataPreProcess


class Window(QMainWindow):

    def __init__(self):
        super().__init__()

        self.origData = None
        self.eny = None
        self.pst = None
        self.corr = None

        self.mainWin = Ui_MainWindow()
        self.mainWin.setupUi(self)

        self.initGui()



        self.mainWin.showTableView.setEditTriggers(QAbstractItemView.NoEditTriggers)

    def initGui(self):
        model = QStandardItemModel()

        for i in range(numTestPoint):
            model.setHorizontalHeaderItem(i, QStandardItem("{}".format(i)))
            model.setVerticalHeaderItem(i, QStandardItem("{}".format(i)))

        self.mainWin.showTableView.setModel(model)

        # self.mainWin.showTextEdit.setText(">>> ")

        self.origData = np.ndarray((numTestPoint, numAntenna, (1+antennaDataIsCplx)))
        self.eny = np.ndarray(numTestPoint)
        self.pst = np.ndarray((numTestPoint, 2))



        for i in range(numTestPoint):
            fileName = DATA_DIR + "{}.txt".format(i)

            data = CsiDataPreProcess.readBestCsiData(fileName)
            self.origData[data["serNum"]] = data["csi"]
            self.eny[data["serNum"]] = data["eny"]
            self.pst[data["serNum"]] = data["pst"]

        self.corr = CsiDataPreProcess.getMatrixCorrelation(self.origData)

        for i in range(numTestPoint):
            for j in range(numTestPoint):
                model.setItem(i, j, QStandardItem("{:.4f}".format(self.corr[i, j])))


    def itemClicked(self, index: QModelIndex):
        row = index.row()
        column = index.column()

        text = self.mainWin.showTextEdit.toPlainText()
        print(text+"<{},{}>".format(row, column))


        string = self.mainWin.showTextEdit.toHtml()

        self.mainWin.showTextEdit.setText(string + "{}:  {:.4f}, {}:  {:.4f} {:.4f}".format(column, self.eny[column], row, self.eny[row], self.corr[row, column]))








if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)

    app.setApplicationName("checkData")

    win = Window()
    win.showMaximized()
    sys.exit(app.exec_())
