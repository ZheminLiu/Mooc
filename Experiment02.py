# coding: utf8
from DataHandler import *
from BiRNNSeqLabel import *

trainRadio = 0.7  # 取80%的数据训练
batchSize = 100  # 每次读取的数据量
epoch = 1  # 迭代次数
drop = 0.6
courseDict = {
    85001: [253007, 357003, 485002, 1001584004],
    20001: [407001, 1001620007],
    21011: [440004],
    43002: [231002, 457001],
    45002: [255006, 419003, 1001596003]
}  # 课程号: [学期号]


def exp(courseId, termId, trainData, testData,
        trainLabel1, testLabel1, trainLabel2, testLabel2,
        trainLabel3, testLabel3, trainLabel4, testLabel4):
    """
    实验1与实验2的模板
    :return:
    """
    #dataClean(trainData, trainLabel1, trainLabel2, trainLabel3, trainLabel4)  # 被排除的数量
    # 数据转换
    trainData, testData = np.array(trainData), np.array(testData)
    trainLabel1, testLabel1 = np.array(trainLabel1), np.array(testLabel1)
    trainLabel2, testLabel2 = np.array(trainLabel2), np.array(testLabel2)
    trainLabel3, testLabel3 = np.array(trainLabel3), np.array(testLabel3)
    trainLabel4, testLabel4 = np.array(trainLabel4), np.array(testLabel4)

    numSteps, size = trainData.shape[1], trainData.shape[2]
    for i in xrange(1, 5):  # 四个定义四次训练
        with tf.variable_scope("%s_%s_%s" % (courseId, termId, i)):
            print "  预测定义%s:" % i
            # 定义tf变量
            data = tf.placeholder(tf.float32, [None, numSteps, size])
            target = tf.placeholder(tf.float32, [None, numSteps, trainLabel1.shape[2]])
            dropout = tf.placeholder(tf.float32)
            session = tf.Session()
            targetLabel = session.run(tf.argmax(eval('testLabel' + str(i)), 2))  # 期望的结果
            sl = BiRNN(data, target, dropout, session)
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
                #zeroMatrix = getZeroMatrix(testData)  # 判断各个周是否要排除
                print "    Epoch:%d," % (e + 1),
                for j in xrange(predict.shape[1]):  # 按周计算正确率
                    #exclude = 0  # 排除的用户数
                    correct = 0  # 预测正确的数量
                    for k in xrange(predict.shape[0]):  # 计算每个用户
                        # if zeroMatrix[k][j] == 1:
                        #     exclude += 1  # 全0,应当排除
                        if predict[k][j] == targetLabel[k][j]:
                            correct += 1
                    accuracy = (correct) * 1.0 / predict.shape[0]
                    print "第%s周的准确率:%3.6f%%" % (j + 1, accuracy * 100),
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
            exp(courseId, termId, trainData, testData,
                trainLabel1, testLabel1, trainLabel2, testLabel2,
                trainLabel3, testLabel3, trainLabel4, testLabel4)
