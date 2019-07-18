#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/9 7:57
# @Author  : ysrs
# @Site    : https://www.zhihu.com/people/ysrs
# @File    : lr_gd.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt


# 实现sigmoid函数，这里需要用到numpy库里的指数函数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# 逻辑回归实现
def logistic_regression(X, Y, W, bias, eta, sample_num, iter_num):
    """
    :param X: 数据集，目前是mat格式，一个样本数据可以有多个特征
    :param Y: 标签，对于二分类问题，我们用1表示正样本，0表示负样本
    :param W: 样本特征值的权重矩阵，也是mat格式
    :param bias: 偏差项
    :param eta: 步长
    :param sample_num: 样本个数
    :param iter_num: 迭代次数
    :return: 最后一次计算的概率值A，权重W，偏差bias，损失函数J
    """
    J = np.zeros((iter_num, 1))
    for i in range(iter_num):
        # 第一步：前向传播
        Z = np.dot(W, X) + bias
        A = sigmoid(Z)
        # 计算代价函数
        J[i] = -(1/sample_num)*(np.dot(Y, np.log(A.T)) + np.dot((1 - Y), np.log((1 - A).T)))
        # 第二步：反向传播
        dZ = A - Y
        dW = (1/sample_num)*np.dot(dZ, X.T)
        db = (1/sample_num)*np.sum(dZ)
        # 第三步：梯度下降
        W = W - eta*dW
        bias = bias - eta*db
    return A, W, bias, J


def plot_best_fit(X, Y, J, W, M, A):
    # 设置显示中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 绘制最终分类图片
    # 根据训练样本标记不同，分为两类不同的点
    x_cord_pos = []      # 正样本的x坐标
    y_cord_pos = []      # 正样本的y坐标
    x_cord_neg = []      # 负样本的x坐标
    y_cord_neg = []      # 负样本的y坐标
    for i in range(M):
        if int(Y[0, i]) == 1:
            x_cord_pos.append(X[0, i])
            y_cord_pos.append(X[1, i])
        else:
            x_cord_neg.append(X[0, i])
            y_cord_neg.append(X[1, i])
    plt.figure('逻辑回归曲线')
    plt.scatter(x_cord_pos, y_cord_pos, c='b', marker='o')
    plt.scatter(x_cord_neg, y_cord_neg, c='r', marker='s')
    x = np.linspace(-3, 3, 100).reshape(100, 1)   # 生成一个x坐标的数组
    y = (-b - W[0, 0] * x) / W[0, 1]              # 根据x坐标计算y值
    plt.plot(x, y, c='y')
    plt.title('逻辑分类结果示意图')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    x_pos = [-1.395634, 0.406704, -2.46015, 0.850433, 1.176813, -0.566606, 0.931635, -0.024205, -0.036453, -0.196949]
    y_pos = [4.662541, 7.067335, 6.866805, 6.920334, 3.16702, 5.749003, 1.589505, 6.151823, 2.690988, 0.444165]

    x_neg = [-0.017612, -0.752157, -1.322371, 0.423363, 0.667394, 0.569411, -0.026632, 1.347183, -1.781871, -0.576525]
    y_neg = [14.053064, 6.53862, 7.152853, 11.054677, 12.741452, 9.548755, 10.427743, 13.1755, 9.097953, 11.778922]

    Y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    x = x_pos + x_neg
    y = y_pos + y_neg

    # 将list格式转换为mat格式
    X = [x, y]
    W = [[0.1, 0.08]]

    X = np.mat(X)
    Y = np.mat(Y)

    # 初始化参数
    feature_num = X.shape[0]                    # 特征数
    sample_num = X.shape[1]                     # 样本个数
    W = np.random.randn(1, feature_num) * 0.01  # 特征值权重
    W = np.mat(W)                               # 特征值权重
    bias = 0                                    # 偏差
    eta = 0.01                                  # 步长
    iter_num = 5000                             # 迭代次数
    A, W, b, J = logistic_regression(X, Y, W, bias, eta, sample_num, iter_num)
    # 将逻辑回归数据结果画图
    plot_best_fit(X, Y, J, W, sample_num, A)
    pass

