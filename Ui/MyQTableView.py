import typing

from PyQt5.QtWidgets import QTableView, QWidget, QMainWindow


class MyQTableView(QTableView):

    def __init__(self, parent, topWin) -> None:
        super().__init__(parent)
        self.top = topWin

        self.clicked.connect(self.top.itemClicked)
