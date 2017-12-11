import numpy as np
import csv


def readData(file):
    data = []
    cnt = []
    with open(file) as csvFile:
        read = csv.reader(csvFile)
        for row in read:
            line = []
            for datnum in row[:-1]:
                try:
                    line = line + [float(datnum)]
                except ValueError:
                    continue
            data.append(line)
            if(row[-1].isdigit()):
                cnt.append([float(row[-1])])
#    print(data)
    return np.mat(data), np.mat(cnt)


# 计算每一列属性值与结果的相关性
def relevant(file):
    data_ori, cnt_ori = readData(file)
    data = data_ori
    cnt = cnt_ori
    y_avg = sum(cnt) / len(cnt)  # 求y的平均值
    cnt = cnt - y_avg
    m, n = np.shape(data)
    corr = []
    var_y = np.dot(cnt.T, cnt) / m
    for i in range(n):
        x_avg = sum(data[:, i]) / len(data[:, i])  # 求x的平均值
        data[:, i] = data[:, i] - x_avg
        var_x = np.dot(data[:, i].T, data[:, i]) / m  # 方差
        cov = np.dot(data[:, i].T, cnt) / m  # 协方差
        temp = cov / (np.sqrt(var_x) * np.sqrt(var_y))  # 相关系数
        corr.append(temp[0, 0])
    index = []
    for i in range(len(corr)):
        if(abs(corr[i]) >= 0.2):
            index.append(i)
    return index


# 数据预处理
def processData(file_train, file):
    data_ori, cnt = readData(file)
    index = relevant(file_train)
    # 剔除相关性太低的列
    m, n = np.shape(data_ori)
    data = np.mat(np.ones((m, 1 + len(index))))  # 添加截距项
    for i in range(len(index)):
        for j in range(m):
            data[j, 1 + i] = data_ori[j, index[i]]
    # min-max归一化
    m, n = np.shape(data)
    for i in range(n):
        min_ = min(data[:, i])
        max_ = max(data[:, i])
        dif = max_ - min_
        if(dif != 0):
            for j in range(m):
                data[j, i] = (data[j, i] - min_) / dif
    return data, cnt


def sigmod(x):
    temp = 1 / (1 + np.exp(-x))
    return np.mat(temp)


def train(file):
    data, cnt = processData(file, file)
    m, n = np.shape(data)
    # 初始化权重，0到1之间的随机数
    w_o = np.mat(np.random.random(size=(1, n * 2)))
    w_h = np.mat(np.random.random(size=(n * 2, n)))
    ratio = 0.001  # 学习率
    for i in range(1000):
        # 初始化权重更新步长
        w_o_change = np.mat(np.zeros((1, n * 2)))
        w_h_change = np.mat(np.zeros((n * 2, n)))
        # 计算最后的输出
        h_in = np.dot(data, w_h.T)
        h_out = sigmod(h_in)
        out_fnl = np.dot(h_out, w_o.T)
        # 计算误差梯度
        err_fnl = cnt - out_fnl
        err_hid = np.multiply(err_fnl * w_o, np.multiply(h_out, (1 - h_out)))
        # 更新权重更新步长
        w_o_change = w_o_change + err_fnl.T * h_out
        w_h_change = w_h_change + err_hid.T * data
        # 更新权重
        w_o = w_o + ratio * w_o_change / m
        w_h = w_h + ratio * w_h_change / m

    return w_o, w_h


def test(file_train, file):
    w_o, w_h = train(file_train)
    data, cnt = processData(file_train, file)

    h_in = np.dot(data, w_h.T)
    h_out = sigmod(h_in)
    out_fnl = np.dot(h_out, w_o.T)

    f = open('15352010_caiye.txt', 'w')
    m, n = np.shape(out_fnl)
    for i in range(m):
        f.write(str(out_fnl[i, 0]) + '\n')
    f.close()


test('BPNN_Dataset/train.csv', 'BPNN_Dataset/test.csv')