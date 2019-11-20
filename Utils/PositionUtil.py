import numpy as np
from numpy import random


class PositionTool:

    def __init__(self) -> None:
        super().__init__()
        self.contents = []

    def read(self, file):
        fr = open(file, 'r')
        contents = fr.readlines()
        fr.close()

        for item in contents:
            # print(item)
            xy = item.split(" ")
            point = Point(int(xy[0]), int(xy[1]))
            self.contents.append(point)

    def __len__(self):
        return len(self.contents)

    def index(self, i):

        point = self.contents[i]
        return "[x:" + str(point.xPos) + " y:" + str(point.yPos) + "]"

    def get(self, numsArr):

        n, m = numsArr.shape
        lengths = np.ndarray((n, m - 1))

        for i in range(n):
            nums = numsArr[i, :-1]
            target = numsArr[i, -1]

            target = self.contents[target]
            length = []
            for j in range(m - 1):
                point = self.contents[nums[j]]
                length.append(point.length(target))

            lengths[i, :] = nums[np.argsort(np.asarray(length))]

        return lengths


class Point:

    def __init__(self, x, y) -> None:
        super().__init__()

        self.xPos = x
        self.yPos = y

    def length(self, point):
        assert type(point) is Point
        return np.sqrt(
            (self.xPos - point.xPos) * (self.xPos - point.xPos) + (self.yPos - point.yPos) * (self.yPos - point.yPos))


class RandomSelection:

    def __init__(self, nTotal, nums, maxNum) -> None:
        super().__init__()
        self.nTotal = nTotal
        self.nums = nums
        self.maxNum = maxNum
        self.contents = np.ndarray((self.nTotal, self.nums))

    def get(self):
        return self.contents

    def launch(self):
        contents = []

        n = self.nTotal

        while n > 0:
            content = []
            m = self.nums
            while m > 0:
                num = random.randint(self.maxNum)
                try:
                    content.index(num)
                except ValueError:
                    content.append(num)
                    m -= 1

            content.sort()

            while True:
                target = random.randint(self.maxNum)
                try:
                    content.index(target)
                except ValueError:
                    content.append(target)
                    break

            try:
                contents.index(content)
            except ValueError:
                contents.append(content)
                n -= 1

                if (n % 100) == 0:
                    print(n)
                # print(n)

        # contents.sort()
        self.contents = np.asarray(contents)
