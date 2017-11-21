import numpy as np
import csv


def readData(file):
    data = []
    label = []
    with open(file) as csvFile:
        read = csv.reader(csvFile)
        for row in read:
            line = [1.0]
            line = line + list(map(float, row[:-1]))
            data.append(line)
            if(row[-1].isdigit()):  # 判断是不是数字，因为测试集是？无法作用为int
                label.append(int(row[-1]))
    return np.mat(data), np.mat(label).T


def logistic(w, x):
    temp = np.dot(x, w)
    temp = 1 / (1 + np.exp(-temp))
    return np.mat(temp)


def process(file):
    data, label = readData(file)
    m, n = np.shape(data)
    weight = np.mat(np.ones((n, 1)))  # 初始化梯度w
    study_ratio = 0.0001  # 初始化学习率
    for iterate in range(5000):  # 更新多次w后再停止
        study_ratio *= 0.995  # 调整学习率
        logi = logistic(weight, data) - label
        detaW = np.dot(logi.T, data)
        weight = weight - study_ratio * detaW.T
    return weight


def predict(train_file, test_file):
    f = open('15352010_caiye.txt', 'w')
    w = process(train_file)
    test_data, test_label = readData(test_file)
    label = logistic(w, test_data)
    for num in label:  # 小于0.5判为0，否则判为1
        num = 0 if num < 0.5 else 1
        f.write(str(num) + '\n')
    f.close()


predict("train.csv", "test.csv")
