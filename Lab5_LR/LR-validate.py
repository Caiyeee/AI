import numpy as np
import csv


def readData(file):
    data = []
    with open(file) as csvFile:
        read = csv.reader(csvFile)
        for row in read:
            line = [1.0]
            line = line + list(map(float, row))
            data.append(line)
    return np.mat(data)


def logistic(w, x):
    temp = np.dot(x, w)
    temp = 1 / (1 + np.exp(-temp))
    return np.mat(temp)


def process(file):
    dataAll = readData(file)
    m, n = np.shape(dataAll[:, :-1])
    for i in range(1):  # 随机打乱次数
        np.random.shuffle(dataAll)  # 随机打乱数据集
        data = dataAll[:int(m * 0.75), :]  # 训练集
        validate = dataAll[int(m * 0.75):m, :]  # 验证集
        weight = np.mat(np.ones((n, 1)))  # 初始化梯度w
        study_ratio = 0.0001  # 学习率
        for iterate in range(5000):
            study_ratio *= 0.995  # 调整学习率
            logi = logistic(weight, data[:, :-1]) - data[:, -1]
            detaW = np.dot(logi.T, data[:, :-1])
            weight = weight - study_ratio * detaW.T
        print(weight)

        label = logistic(weight, validate[:, :-1])

        yes = 0  # 预测正确的个数
        for j in range(len(label)):  # 和真正的标签差值小于0.5则预测正确
            if abs(label[j] - validate[j, -1]) < 0.5:
                yes += 1
        print("随机" + str(i) + "准确率: ", yes / len(label))


process("train.csv")
