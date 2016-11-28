# coding: utf8
from DataHandler import *
from SequenceLabeling import *
trainRadio = 0.8  # 取80%的数据训练
batchSize = 100  # 每次读取的数据量
drop = 0.5
courseDict = {
    20001: [407001, 1001620007],
    21011: [440004],
    43002: [231002, 457001],
    45002: [255006, 419003, 1001596003],
    85001: [253007, 357003, 485002, 1001584004]
}  # 课程号: [学期号]
def exp1():
    """
    实验1,每个学期各自训练
    :return:
    """
    for courseId in courseDict:
        for termId in courseDict[courseId]:
            # 读取数据
            print "读取%s_%s的数据" % (courseId, termId)
            data, label1, label2, label3, label4 = readData(courseId, termId)
            trainNums = int(len(data)*trainRadio)  # 训练数量
            trainData, testData = data[:trainNums], data[trainNums:]
            trainLabel1, testLabel1 = label1[:trainNums], label1[trainNums:]
            trainLabel2, testLabel2 = label2[:trainNums], label2[trainNums:]
            trainLabel3, testLabel3 = label3[:trainNums], label3[trainNums:]
            trainLabel4, testLabel4 = label4[:trainNums], label4[trainNums:]
            numSteps, size = data.shape[1], data.shape[2]
            for i in xrange(1, 5):  # 四个定义四次训练
                with tf.variable_scope("%s_%s_%s" % (courseId, termId, i)):
                    print "  开始定义%s的训练" % i
                    data = tf.placeholder(tf.float32, [None, numSteps, size])
                    target = tf.placeholder(tf.float32, [None, numSteps, label1.shape[2]])
                    dropout = tf.placeholder(tf.float32)
                    sl = SequenceLabelling(data, target, dropout)
                    session = tf.Session()
                    session.run(tf.initialize_all_variables())
                    for j in xrange(int(np.ceil(trainData.shape[0]/batchSize))):
                        batchData = trainData[j*batchSize:(j+1)*batchSize]
                        batchTarget = eval('trainLabel'+str(i))[j*batchSize:(j+1)*batchSize]
                        session.run(sl.optimize, {data: batchData, target: batchTarget, dropout: drop})
                    error = session.run(sl.error, {data: testData, target: eval('testLabel'+str(i)), dropout: drop})
                    print "  Accuracy: {:3.2f}%".format(100*error)
exp1()
