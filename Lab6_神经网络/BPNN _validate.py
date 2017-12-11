import numpy as np
import csv


def readData(file):
    data = []
    with open(file) as csvFile:
        read = csv.reader(csvFile)
        for row in read:
            line = []
            for datnum in row:
                try:
                    line = line + [float(datnum)]
                except ValueError:
                    continue
            data.append(line)
#    print(data)
    return np.mat(data)


def relevant(file):
    data_ori = readData(file)
    data = data_ori
    cnt = data_ori[:, -1]
    y_avg = sum(cnt) / len(cnt)
    cnt = cnt - y_avg
    m, n = np.shape(data)
    corr = []
    var_y = np.dot(cnt.T, cnt) / m
    for i in range(n - 1):
        x_avg = sum(data[:, i]) / len(data[:, i])
        data[:, i] = data[:, i] - x_avg
        var_x = np.dot(data[:, i].T, data[:, i]) / m
        cov = np.dot(data[:, i].T, cnt) / m
        temp = cov / (np.sqrt(var_x) * np.sqrt(var_y))
        corr.append(temp[0, 0])
    index = []
    for i in range(len(corr)):
        if(abs(corr[i]) >= 0.2):
            index.append(i)
#    print(index)
    return index


def processData(file_train, file):
    data_ori = readData(file)
    index = relevant(file_train)
    # 剔除相关性太低的列
    m, n = np.shape(data_ori)
    data = np.mat(np.ones((m, 2 + len(index))))
    for i in range(len(index)):
        for j in range(m):
            data[j, 1 + i] = data_ori[j, index[i]]
    for i in range(m):
        data[j, -1] = data_ori[j, -1]
    # min-max归一化
    m, n = np.shape(data)
    for i in range(n):
        min_ = min(data[:, i])
        max_ = max(data[:, i])
        dif = max_ - min_
        if(dif != 0):
            for j in range(m):
                data[j, i] = (data[j, i] - min_) / dif
    return data


def sigmod(x):
    temp = 1 / (1 + np.exp(-x))
    return np.mat(temp)


def train(file):
    dataAll = processData(file, file)
    m, n = np.shape(dataAll)
    np.random.shuffle(dataAll)  # 打乱正数据集
    data = dataAll[:int(m * 0.75), :]  # 训练集
    validate = dataAll[int(m * 0.75):m, :]  # 验证集
    cnt_train = data[:, -1]  # 训练集的标签
    cnt_valid = validate[:, -1]  # 验证集的标签
    data = data[:, :-1]  # 训练集的特征向量
    validate = validate[:, :-1]  # 验证集的特征向量
    m, n = np.shape(data)
    w_o = np.mat(np.random.random(size=(1, n * 2)))
    w_h = np.mat(np.random.random(size=(n * 2, n)))
    ratio = 0.001
    loss = open('loss.csv', 'w')

    for i in range(3000):
        w_o_change = np.mat(np.zeros((1, n * 2)))
        w_h_change = np.mat(np.zeros((n * 2, n)))

        h_in = np.dot(data, w_h.T)
        h_out = sigmod(h_in)
        out_fnl = np.dot(h_out, w_o.T)

        err_fnl = cnt_train - out_fnl
        err_hid = np.multiply(err_fnl * w_o, np.multiply(h_out, (1 - h_out)))
        # 计算训练集的mse
        p, q = np.shape(err_fnl)
        sum = 0.0
        for j in range(p):
            sum = sum + err_fnl[j, 0] * err_fnl[j, 0]
        sum = sum / p
        loss.write(str(sum) + ',')

        w_o_change = w_o_change + err_fnl.T * h_out
        w_h_change = w_h_change + err_hid.T * data

        w_o = w_o + ratio * w_o_change / m
        w_h = w_h + ratio * w_h_change / m

        # 对验证集进行预测
        h_in = np.dot(validate, w_h.T)
        h_out = sigmod(h_in)
        out_fnl = np.dot(h_out, w_o.T)
        err_val = cnt_valid - out_fnl
        # 计算验证集的mse
        p, q = np.shape(err_val)
        mse = 0.0
        for j in range(p):
            mse = mse + err_val[j, 0] * err_val[j, 0]
        mse = mse / p
        loss.write(str(mse) + '\n')
    loss.close()

    # 用最后模型对验证集进行预测
    h_in = np.dot(validate, w_h.T)
    h_out = sigmod(h_in)
    out_fnl = np.dot(h_out, w_o.T)
    err_val = cnt_valid - out_fnl
    p, q = np.shape(err_val)


train("BPNN_Dataset/train.csv")
