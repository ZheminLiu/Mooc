# coding: utf8
import numpy as np


def readData(courseId, termId):
    """
    根据课程号与学期号，读取文件
    :param courseId: 课程号
    :param termId: 学期号
    :return: (数据，定义1的oneHot，定义2的，定义3的，定义4的）,shape = (用户数，周数，特征数）
    """
    dataFileName = "dataFiles/data_%s_%s" % (courseId, termId)
    labelFileName = "dataFiles/label_%s_%s" % (courseId, termId)
    data = []  # 数据
    label1, label2, label3, label4 = [], [], [], []  # 四个定义
    with open(dataFileName, 'r') as dataFile:
        for line in dataFile:
            lineArray = line.strip().split("\t")[1:]  # 每一项为一个周的数据
            data.append([map(int, week.split()) for week in lineArray])
    with open(labelFileName, 'r') as labelFile:
        for line in labelFile:
            lineArray = line.strip().split("\t")[1:]  # 每一项为一个周的四个定义
            weeks = [map(int, week.split()) for week in lineArray]
            label1.append([[week[0], 1 - week[0]] for week in weeks])
            label2.append([[week[1], 1 - week[1]] for week in weeks])
            label3.append([[week[2], 1 - week[2]] for week in weeks])
            label4.append([[week[3], 1 - week[3]] for week in weeks])
    return np.array(data), np.array(label1), np.array(label2), np.array(label3), np.array(label4)
