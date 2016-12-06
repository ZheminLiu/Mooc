# coding: utf8
from DataHandler import *
from SequenceLabeling import *

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
            zeros = dataClean(trainData, testData, (trainLabel1, testLabel1),
                              (trainLabel2, testLabel2),
                              (trainLabel3, testLabel3),
                              (trainLabel4, testLabel4))  # 被排除的数量

            trainData, testData = np.array(trainData), np.array(testData)
            trainLabel1, testLabel1 = np.array(trainLabel1), np.array(testLabel1)
            trainLabel2, testLabel2 = np.array(trainLabel2), np.array(testLabel2)
            trainLabel3, testLabel3 = np.array(trainLabel3), np.array(testLabel3)
            trainLabel4, testLabel4 = np.array(trainLabel4), np.array(testLabel4)

            numSteps, size = trainData.shape[1], trainData.shape[2]
            for i in xrange(1, 5):  # 四个定义四次训练
                with tf.variable_scope("%s_%s_%s" % (courseId, termId, i)):
                    print "  预测定义%s:" % i
                    data = tf.placeholder(tf.float32, [None, numSteps, size])
                    target = tf.placeholder(tf.float32, [None, numSteps, trainLabel1.shape[2]])
                    dropout = tf.placeholder(tf.float32)
                    session = tf.Session()
                    targetLabel = session.run(tf.argmax(eval('testLabel' + str(i)), 2)).T  # 期望的结果
                    sl = SequenceLabelling(data, target, dropout, session)
                    session.run(tf.initialize_all_variables())
                    for e in xrange(epoch):
                        for j in xrange(int(np.ceil(trainData.shape[0] / batchSize))):
                            batchData = trainData[j * batchSize:(j + 1) * batchSize]
                            batchTarget = eval('trainLabel' + str(i))[j * batchSize:(j + 1) * batchSize]
                            session.run(sl.optimize, {data: batchData, target: batchTarget, dropout: drop})
                        predict = session.run(sl.prediction, {data: testData, target: eval('testLabel' + str(i)),
                                                              dropout: drop})
                        predict = session.run(tf.argmax(predict, 2)).T  # 预测的结果
                        error = session.run(sl.error,
                                            {data: testData, target: eval('testLabel' + str(i)), dropout: drop})
                        accuracy = (zeros + len(testData) * error) * 1.0 / (len(testData) + zeros)
                        print "    Epoch{:d}, 预测的正确率:{:3.2f}%,".format(e + 1, 100 * accuracy),
                        for j in xrange(len(predict)):  # 输出各周的准确率
                            weekAccuracy = session.run(tf.reduce_mean(
                                tf.cast(tf.equal(predict[j], targetLabel[j]), tf.float32)))
                            weekAccuracy = (zeros + len(testData) * weekAccuracy) * 1.0 / \
                                           (len(testData) + zeros)
                            print "第%s周的准确率:%3.2f%%," % (j + 1, weekAccuracy * 100),
                        print


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

            zeros = dataClean(trainData, testData, (trainLabel1, testLabel1),
                              (trainLabel2, testLabel2),
                              (trainLabel3, testLabel3),
                              (trainLabel4, testLabel4))  # 被排除的数量

            trainData, testData = np.array(trainData), np.array(testData)
            trainLabel1, testLabel1 = np.array(trainLabel1), np.array(testLabel1)
            trainLabel2, testLabel2 = np.array(trainLabel2), np.array(testLabel2)
            trainLabel3, testLabel3 = np.array(trainLabel3), np.array(testLabel3)
            trainLabel4, testLabel4 = np.array(trainLabel4), np.array(testLabel4)

            # 数据转换
            print "  已进行周数的转换"
            modelNum = trainData.shape[1]
            testData = dataConvert(modelNum, testData)
            testLabel1 = dataConvert(modelNum, testLabel1)
            testLabel2 = dataConvert(modelNum, testLabel2)
            testLabel3 = dataConvert(modelNum, testLabel3)
            testLabel4 = dataConvert(modelNum, testLabel4)

            numSteps, size = trainData.shape[1], trainData.shape[2]
            for i in xrange(1, 5):  # 四个定义四次训练
                with tf.variable_scope("%s_%s" % (courseId, i)):
                    print "  预测定义%s:" % i
                    data = tf.placeholder(tf.float32, [None, numSteps, size])
                    target = tf.placeholder(tf.float32, [None, numSteps, trainLabel1.shape[2]])
                    dropout = tf.placeholder(tf.float32)
                    session = tf.Session()
                    sl = SequenceLabelling(data, target, dropout, session)
                    targetLabel = session.run(tf.argmax(eval('testLabel' + str(i)), 2)).T  # 期望的结果
                    session.run(tf.initialize_all_variables())
                    for e in xrange(epoch):
                        for j in xrange(int(np.ceil(trainData.shape[0] / batchSize))):
                            batchData = trainData[j * batchSize:(j + 1) * batchSize]
                            batchTarget = eval('trainLabel' + str(i))[j * batchSize:(j + 1) * batchSize]
                            session.run(sl.optimize, {data: batchData, target: batchTarget, dropout: drop})
                        predict = session.run(sl.prediction, {data: testData, target: eval('testLabel' + str(i)),
                                                              dropout: drop})
                        predict = session.run(tf.argmax(predict, 2)).T  # 预测的结果
                        error = session.run(sl.error,
                                            {data: testData, target: eval('testLabel' + str(i)), dropout: drop})
                        accuracy = (zeros + len(testData) * error) * 1.0 / (len(testData) + zeros)
                        print "    Epoch{:d}, 预测的总正确率:{:3.2f}%,".format(e + 1, 100 * accuracy),
                        for j in xrange(len(predict)):  # 输出各周的准确率
                            weekAccuracy = session.run(tf.reduce_mean(
                                tf.cast(tf.equal(predict[j], targetLabel[j]), tf.float32)))
                            weekAccuracy = (zeros + len(testData) * weekAccuracy) * 1.0 / \
                                           (len(testData) + zeros)
                            print "第%s周的准确率:%3.2f%%," % (j + 1, weekAccuracy * 100),
                        print
