import os
import shutil


class FileOprter:


    # 判断文件或文件夹是否存在

    @staticmethod
    def exist(path):
        return os.path.exists(path)

    # 判断是否为文件 True:是个文件 False:文件夹或不存在
    @staticmethod
    def isFile(file):
        return os.path.isfile(file)

    # 判断是否为文件夹 True:是个文件夹 False:文件或不存在
    @staticmethod
    def isDir(path):
        return os.path.isdir(path)


    @staticmethod
    def __doubleCheck(srcFile, dstFile, comment=None):

        if comment is None:
            info = ""
        else:
            info = comment

        if not FileOprter.exist(srcFile):
            info = info + "失败!源文件：" + srcFile + " 不存在"
            print(info)
            return False

        if FileOprter.exist(dstFile) and srcFile != dstFile:
            info = info + "失败!目标文件已存在或是个文件夹,path:" + dstFile
            print(info)
            return False

        return True

    # 文件重命名
    @staticmethod
    def rename(srcFile, dstFile):
        if not FileOprter.__doubleCheck(srcFile, dstFile, comment="重命名文件"):
            return
        os.rename(srcFile, dstFile)

    # 移动文件
    @staticmethod
    def move(srcFile, dstFile):
        if not FileOprter.__doubleCheck(srcFile, dstFile, comment="移动文件"):
            return

        shutil.move(srcFile, dstFile)

    # 复制文件
    @staticmethod
    def copy(srcFile, dstFile):
        if not FileOprter.__doubleCheck(srcFile, dstFile, comment="复制文件"):
            return

        shutil.copy(srcFile, dstFile)

    @staticmethod
    def files(path, postFix=[]):

        try:
            flag = True
            assert type(postFix) is list
            if len(postFix) == 0:
                flag = False
        except ValueError:
            assert type(postFix) is str
            postFix = [].append(postFix)

        if not FileOprter.isDir(path):
            print("获取目录所有文件失败，目录:"+path+"不存在")
            return

        allFiles = []
        allFullPathFiles = []

        for root, subDirs, files in os.walk(path):
            break

        if not root.endswith("/"):
            root = root + "/"

        for file in files:
            suffix = file.split(".")[-1]

            if not flag:
                allFiles.append(file)
                allFullPathFiles.append(root + file)
            else:
                if suffix in postFix:
                    allFiles.append(file)
                    allFullPathFiles.append(root + file)

        return allFullPathFiles, allFiles



    # 移动文件夹下所有文件到目标文件夹
    @staticmethod
    def moveDir(srcPath, dstPath):

        assert type(srcPath) is str
        assert type(dstPath) is str

        if not srcPath.endswith("/"):
            srcPath = srcPath + "/"

        if not dstPath.endswith("/"):
            dstPath = dstPath + "/"

        if FileOprter.isDir(srcPath) and FileOprter.isDir(dstPath):
            _, files = FileOprter.files(srcPath)

            n = len(files)

            for i in range(n):
                srcFile = srcPath + files[i]
                dstFile = dstPath + files[i]

                FileOprter.move(srcFile, dstFile)

    # 拷贝文件夹下所有文件到目标文件夹
    @staticmethod
    def copyDir(srcPath, dstPath):
        assert type(srcPath) is str
        assert type(dstPath) is str

        if not srcPath.endswith("/"):
            srcPath = srcPath + "/"

        if not dstPath.endswith("/"):
            dstPath = dstPath + "/"

        if FileOprter.isDir(srcPath) and FileOprter.isDir(dstPath):
            _, files = FileOprter.files(srcPath)


            n = len(files)
            for i in range(n):
                srcFile = srcPath + files[i]
                dstFile = dstPath + files[i]

                FileOprter.copy(srcFile, dstFile)

    # 删除文件或文件夹
    @staticmethod
    def delete(file):
        if FileOprter.isFile(file):
            os.remove(file)
        elif FileOprter.isDir(file):
            shutil.rmtree(file)

    @staticmethod
    def rename(srcFile, dstFile):
        if not FileOprter.__doubleCheck(srcFile, dstFile, comment="重命名文件"):
            return
        os.rename(srcFile, dstFile)

#
# dir1 = "/home/lxw/PycharmProjects/TerminalLocalization/datas/fortest/"
#
# n = 2640
#
# for i in range(n):
#     num = i + 1
#     srcFile = dir1 + str(num) + ".txt"
#
#     if (num % 10) == 0:
#         name = str((num//10)) + "_10.txt"
#     else:
#         name = str((num//10) + 1) + "_" + str(num % 10) + ".txt"
#
#     dstFile = dir1 + name
#
#     FileOprter.rename(srcFile, dstFile)





