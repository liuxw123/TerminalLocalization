from Utils.ReadData import ReadData
from Utils.DataOprt import DataOprt
from Utils.PositionUtil import PositionTool,Point
from Utils.PositionUtil import RandomSelection

import numpy as np
import matplotlib.pyplot as plt

filePath = "/home/lxw/PycharmProjects/TerminalLocalization/datas/fortest/"
rd = ReadData()



data = rd.getData(filePath)

# rd.printArr(data['absSum'], numPos=3, title="absSum")

# rd.printMatrix(data['chn'], numPos=3, title="CSI")

# rd.printMatrix(DataOprt.correlation(data['chn']), numPos=4, title="相关性")



# file = "/home/lxw/PycharmProjects/TerminalLocalization/datas/position.txt"
#
# tool = PositionTool()
# tool.read(file)
#
# positions = tool.contents
#
# rs = RandomSelection(1000, 20, 264)
#
# rs.launch()
#
# res = rs.get()
# pt = PositionTool()
# pt.read(file)
# ans = pt.get(res)


#
# file = "/home/lxw/PycharmProjects/TerminalLocalization/datas/position.txt"
#
# tool = PositionTool()
# tool.read(file)
#
# positions = tool.contents
#
# xPos = np.ndarray(len(positions))
# yPos = np.ndarray(len(positions))
#
# for i, point in enumerate(positions):
#     assert type(point) is Point
#     xPos[i] = point.xPos
#     yPos[i] = point.yPos
#
# plt.scatter(xPos, yPos, marker="x")
# plt.show()



