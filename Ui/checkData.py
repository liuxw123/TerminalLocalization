# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'checkData.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

from Ui.MyQTableView import MyQTableView

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout.setObjectName("verticalLayout")
        self.showTableView = MyQTableView(self.tab, MainWindow)
        self.showTableView.setObjectName("showTableView")
        self.verticalLayout.addWidget(self.showTableView)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")
        self.horizontalLayout.addWidget(self.tabWidget)
        self.tabWidget_2 = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.tab_3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.showTextEdit = QtWidgets.QTextEdit(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.showTextEdit.setFont(font)
        self.showTextEdit.setObjectName("showTextEdit")
        self.horizontalLayout_3.addWidget(self.showTextEdit)
        self.tabWidget_2.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.tab_4)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.bkpTextEdit = QtWidgets.QTextEdit(self.tab_4)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.bkpTextEdit.setFont(font)
        self.bkpTextEdit.setObjectName("bkpTextEdit")
        self.horizontalLayout_2.addWidget(self.bkpTextEdit)
        self.tabWidget_2.addTab(self.tab_4, "")
        self.horizontalLayout.addWidget(self.tabWidget_2)
        self.horizontalLayout.setStretch(0, 4)
        self.horizontalLayout.setStretch(1, 1)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 28))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        self.tabWidget_2.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "相关系数"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "坐标图"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_3), _translate("MainWindow", "打印相关信息"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_4), _translate("MainWindow", "命令"))

