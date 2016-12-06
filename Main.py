# coding: utf8
from DataHandler import *
from DynamicSeqLabeling import *

trainRadio = 0.8  # 取80%的数据训练
batchSize = 100  # 每次读取的数据量
epoch = 30  # 迭代次数
drop = 0.5
courseDict = {
    43002: [231002, 457001],
    85001: [253007, 357003, 485002, 1001584004],
    20001: [407001, 1001620007],
    21011: [440004],
    45002: [255006, 419003, 1001596003]
}  # 课程号: [学期号]


def exp3():
    """
    实验3.动态输出每周的测试结果
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
                    model = SequenceLabelling(data, target, dropout, session)
                    # 参数变量初始化
                    session.run(tf.initialize_all_variables())

                    # 迭代进行模型的训练
                    for _ in xrange(epoch):
                        for j in xrange(int(np.ceil(trainData.shape[0] / batchSize))):
                            batch_data = trainData[j * batchSize:(j + 1) * batchSize]
                            batch_target = eval('trainLabel' + str(i))[j * batchSize:(j + 1) * batchSize]
                            session.run(model.optimize, {data: batch_data, target: batch_target, dropout: drop})
                            train_accuracy = session.run(model.correct,
                                                         {data: batch_data, target: batch_target, dropout: 1.0})
                        if _ % 10 == 0:
                            print "     Epoch{:d}: 训练正确率为:{:3.2f}%".format(_ + 1, train_accuracy*100)

                    # 预测模型的评估
                    correct = session.run(model.dynamic_correct,
                                          {data: testData, target: eval('testLabel' + str(i)),
                                           dropout: 1.0})
                    accuracy = session.run(model.correct,
                                           {data: testData, target: eval('testLabel' + str(i)),
                                            dropout: 1.0})

                    # 动态地输出每周的预测正确率
                    for k in xrange(len(correct)):
                        correct[k] = '%.*f' % (
                        2, (zeros + len(testData) * correct[k]) * 1.0 / (len(testData) + zeros) * 100)
                        # 计算全局平均的预测正确率
                    accuracy = (zeros + len(testData) * accuracy) * 1.0 / (len(testData) + zeros)
                    print "         评估每周预测正确率:" + ", ".join(correct)
                    print "         评估平均预测正确率:{:3.2f}".format(accuracy * 100)


exp3()
