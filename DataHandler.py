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
    return data, label1, label2, label3, label4


def dataConvert(modelNum, testData):
    """
    对于训练模型与测试数据长度不匹配的情况，进行数据转化
    :param modelNum: 训练模型的长度
    :param testData: 测试数据（ndarray)
    :return: 转换后的测试数据
    """
    r = modelNum * 1.0 / testData.shape[1]  # 转换的比例因子
    users, features = testData.shape[0], testData.shape[2]
    result = np.zeros((users, modelNum, features))
    for user in xrange(users):
        # 此处倒序映射，如果是需要扩充，由于result默认全0,则未映射到的部分已经补0
        # 如果是压缩，当有多个周映射到同一周时，倒序会自动取靠前的周的数据
        for week in xrange(testData.shape[1] - 1, -1, -1):
            newWeek = int(np.round((week + 1) * r)) - 1  # 化为以1开始，然后转换，四舍五入，再减1
            if newWeek >= 0:
                result[user][newWeek] = testData[user][week]  # 映射
    return result


def dataClean(trainData, testData, *trainAndTestLabels):
    """
    对全0数据进行清除，trainAndTestLabels为(trainLabel, testLabel)数组
    :param trainData:
    :param testData:
    :param trainAndTestLabels:
    :return:
    """
    trainZeros, testZeros = [], []  # 被排除索引
    for i in xrange(len(trainData)):
        if sum(sum(np.array(trainData[i]))) == 0:
            trainZeros.append(i)
    for i in xrange(len(testData)):
        if sum(sum(np.array(testData[i]))) == 0:
            testZeros.append(i)
    for i in xrange(len(trainZeros)):
        trainNum = trainZeros[i] - i
        trainData.pop(trainNum)
        for trainLabel, testLabel in trainAndTestLabels:
            trainLabel.pop(trainNum)
    for i in xrange(len(testZeros)):
        testNum = testZeros[i] - i
        testData.pop(testNum)
        for trainLabel, testLabel in trainAndTestLabels:
            testLabel.pop(testNum)
    return len(testZeros)
