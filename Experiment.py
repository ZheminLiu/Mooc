# coding: utf8
from DataHandler import *
from SequenceLabeling import *
from BiRNNSeqLabel import *

trainRadio = 0.8  # 取80%的数据训练
batchSize = 100  # 每次读取的数据量
epoch = 1  # 迭代次数
drop = 0.5
courseDict = {
    85001: [253007, 357003, 485002, 1001584004],
    20001: [407001, 1001620007],
    21011: [440004],
    43002: [231002, 457001],
    45002: [255006, 419003, 1001596003]
}  # 课程号: [学期号]


def run(courseId, termId, expNum, model, trainData, testData,
        trainLabel1, testLabel1, trainLabel2, testLabel2,
        trainLabel3, testLabel3, trainLabel4, testLabel4):
    """
    实验1与实验2的模板
    :param expNum 实验编号
    :param model 使用的模型
    :return:
    """
    #dataClean(trainData, trainLabel1, trainLabel2, trainLabel3, trainLabel4)  # 被排除的数量
    # 数据转换

    numSteps, size = trainData.shape[1], trainData.shape[2]
    for i in xrange(1, 5):  # 四个定义四次训练
        with tf.variable_scope("%s_%s_%s_%s" % (courseId, termId, expNum, i)):
            print "  预测定义%s:" % i
            # 定义tf变量
            data = tf.placeholder(tf.float32, [None, numSteps, size])
            target = tf.placeholder(tf.float32, [None, numSteps, trainLabel1.shape[2]])
            dropout = tf.placeholder(tf.float32)
            session = tf.Session()
            targetLabel = session.run(tf.argmax(eval('testLabel' + str(i)), 2))  # 期望的结果
            sl = model(data, target, dropout, session)
            # session.run(tf.global_variables_initializer())
            session.run(tf.initialize_all_variables())
            for e in xrange(epoch):
                # 迭代训练
                for j in xrange(int(np.ceil(trainData.shape[0] / batchSize))):
                    batchData = trainData[j * batchSize:(j + 1) * batchSize]
                    batchTarget = eval('trainLabel' + str(i))[j * batchSize:(j + 1) * batchSize]
                    session.run(sl.optimize, {data: batchData, target: batchTarget, dropout: drop})
                predict = session.run(sl.prediction, {data: testData, target: eval('testLabel' + str(i)),
                                                      dropout: drop})
                # 预测的结果，shape = (用户数量，周数），值为0/1
                predict = session.run(tf.argmax(predict, 2))
                print "    Epoch:%d," % (e + 1),
                for j in xrange(predict.shape[1]):  # 按周计算正确率
                    correct = 0.0  # 预测正确的数量
                    for k in xrange(predict.shape[0]):  # 计算每个用户
                        if predict[k][j] == targetLabel[k][j]:
                            correct += 1
                    accuracy = correct / predict.shape[0]
                    print "第%s周的准确率:%3.2f%%" % (j + 1, accuracy * 100),
                print


def exp1():
    """
    实验1,每个学期各自训练
    :return:
    """
    for courseId in courseDict:
        for termId in courseDict[courseId]:
            # 读取数据
            print "课程号:%s, 学期号:%s:" % (courseId, termId)
            data, label1, label2, label3, label4 = readData(courseId, termId)
            trainNums = int(len(data) * trainRadio)  # 训练数量
            trainData, testData = data[:trainNums], data[trainNums:]
            trainLabel1, testLabel1 = label1[:trainNums], label1[trainNums:]
            trainLabel2, testLabel2 = label2[:trainNums], label2[trainNums:]
            trainLabel3, testLabel3 = label3[:trainNums], label3[trainNums:]
            trainLabel4, testLabel4 = label4[:trainNums], label4[trainNums:]
            run(courseId, termId, 1, SequenceLabelling, trainData, testData,
                trainLabel1, testLabel1, trainLabel2, testLabel2,
                trainLabel3, testLabel3, trainLabel4, testLabel4)


def exp2():
    """
    实验2,第一学期为训练数据，第二学期为测试数据
    :return:
    """
    for courseId in courseDict:
        if len(courseDict[courseId]) > 1:  # 学期数大于2
            trainTermId, testTermId = courseDict[courseId][0], courseDict[courseId][1]
            print "课程号:%s, 训练学期号:%s, 测试学期号%s:" % (courseId, trainTermId, testTermId)
            trainData, trainLabel1, trainLabel2, trainLabel3, trainLabel4 = readData(courseId, trainTermId)
            testData, testLabel1, testLabel2, testLabel3, testLabel4 = readData(courseId, testTermId)

            # # 数据转换
            # print "  已进行周数的转换"
            # modelNum = trainData.shape[1]
            # testData = dataConvert(modelNum, testData)
            # testLabel1 = dataConvert(modelNum, testLabel1)
            # testLabel2 = dataConvert(modelNum, testLabel2)
            # testLabel3 = dataConvert(modelNum, testLabel3)
            # testLabel4 = dataConvert(modelNum, testLabel4)
            run(courseId, trainTermId, 2, SequenceLabelling, trainData, testData,
                trainLabel1, testLabel1, trainLabel2, testLabel2,
                trainLabel3, testLabel3, trainLabel4, testLabel4)

def exp3():
    """
    使用BiRNN的实验1
    :return:
    """
    for courseId in courseDict:
        for termId in courseDict[courseId]:
            # 读取数据
            print "课程号:%s, 学期号:%s:" % (courseId, termId)
            data, label1, label2, label3, label4 = readData(courseId, termId)
            trainNums = int(len(data) * trainRadio)  # 训练数量
            trainData, testData = data[:trainNums], data[trainNums:]
            trainLabel1, testLabel1 = label1[:trainNums], label1[trainNums:]
            trainLabel2, testLabel2 = label2[:trainNums], label2[trainNums:]
            trainLabel3, testLabel3 = label3[:trainNums], label3[trainNums:]
            trainLabel4, testLabel4 = label4[:trainNums], label4[trainNums:]
            run(courseId, termId, 3, BiRNN, trainData, testData,
                trainLabel1, testLabel1, trainLabel2, testLabel2,
                trainLabel3, testLabel3, trainLabel4, testLabel4)


def exp4():
    """
    使用Bi的实验2
    :return:
    """
    for courseId in courseDict:
        if len(courseDict[courseId]) > 1:  # 学期数大于2
            trainTermId, testTermId = courseDict[courseId][0], courseDict[courseId][1]
            print "课程号:%s, 训练学期号:%s, 测试学期号%s:" % (courseId, trainTermId, testTermId)
            trainData, trainLabel1, trainLabel2, trainLabel3, trainLabel4 = readData(courseId, trainTermId)
            testData, testLabel1, testLabel2, testLabel3, testLabel4 = readData(courseId, testTermId)

            run(courseId, trainTermId, 4, BiRNN, trainData, testData,
                trainLabel1, testLabel1, trainLabel2, testLabel2,
                trainLabel3, testLabel3, trainLabel4, testLabel4)

def exp5():
    """
    课程间的训练与测试，两两组合
    :return:
    """
    courseList = courseDict.keys()  # 课程列表
    for i in xrange(len(courseList)):  # 两两组合
        for j in xrange(len(courseList)):
            if i != j:
                trainCourseId, testCourseId = courseList[i], courseList[j]
                trainTermId, testTermId = courseDict[trainCourseId][0], courseDict[testCourseId][0]
                trainData, trainLabel1, trainLabel2, trainLabel3, trainLabel4 = readData(trainCourseId, trainTermId)
                testData, testLabel1, testLabel2, testLabel3, testLabel4 = readData(testCourseId, testTermId)
                print u"训练课程号:%s, 学期号:%s, 周数:%s; 测试课程号:%s, 学期号:%s, 周数:%s" \
                    % (trainCourseId, trainTermId, trainData.shape[1], testCourseId,
                       testTermId, testData.shape[1])
                modelNum = trainData.shape[1]
                testData = dataConvert(modelNum, testData)
                testLabel1 = dataConvert(modelNum, testLabel1)
                testLabel2 = dataConvert(modelNum, testLabel2)
                testLabel3 = dataConvert(modelNum, testLabel3)
                testLabel4 = dataConvert(modelNum, testLabel4)
                run(str(trainCourseId)+str(testCourseId), trainTermId, 5, BiRNN, trainData, testData,
                    trainLabel1, testLabel1, trainLabel2, testLabel2,
                    trainLabel3, testLabel3, trainLabel4, testLabel4)
