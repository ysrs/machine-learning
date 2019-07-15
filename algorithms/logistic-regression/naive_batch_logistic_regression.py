#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/9 7:57
# @Author  : ysrs
# @Site    : https://www.zhihu.com/people/ysrs
# @File    : naive_batch_logistic_regression.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt


# 实现sigmoid函数，这里需要用到numpy库里的指数函数
def naive_sigmoid(list_x):
    m_x = len(list_x)
    n_x = len(list_x[0])
    res_list = []
    for i in range(m_x):
        temp_list = []
        for j in range(n_x):
            res = 1.0 / (1.0 + np.exp(-list_x[i][j]))
            temp_list.append(res)
        res_list.append(list(temp_list))
    return res_list


# 这样效率比较低，但是为了更直观演示，我们还是使用list存储数据来进行计算
def naive_matrix_dot(list_a=[], list_b=[]):
    m_a = len(list_a)
    n_a = len(list_a[0])
    m_b = len(list_b)
    n_b = len(list_b[0])
    res_list = []
    for i in range(m_a):
        temp_list = []
        for j in range(n_b):
            res = 0.0
            for temp_i in range(n_a):
                res += (list_a[i][temp_i] * list_b[temp_i][j])
            temp_list.append(res)
        res_list.append(temp_list)
    return res_list


def naive_matrix_add(list_a=[], b=0.0):
    """
    :param list_a: 二维数组用于表示矩阵，第一维表示矩阵行，第二维表示矩阵列
    :param b: 浮点型数值，用于和矩阵中的每一个元素做减法
    :return: 数组中的每一个元素与浮点数值b相减后，产生的新的数组
    """
    m = len(list_a)
    n = len(list_a[0])
    res_list = []
    for i in range(m):
        temp_list = []
        for j in range(n):
            temp_list.append(list_a[i][j] + b)
        res_list.append(temp_list)
    return res_list


def naive_matrix_sub(list_a=[], list_b=[]):
    m = len(list_a)
    n = len(list_a[0])
    res_list = []
    for i in range(m):
        temp_list = []
        for j in range(n):
            temp_list.append(list_a[i][j] - list_b[i][j])
        res_list.append(temp_list)
    return res_list


def naive_matrix_transpose(list_x=[]):
    m_x = len(list_x)
    n_x = len(list_x[0])
    res_list = []
    for i in range(n_x):
        temp_list = []
        for j in range(m_x):
            temp_list.append(list_x[j][i])
        res_list.append(temp_list)
    return res_list


def naive_matrix_product(list_x=[], a=0.0):
    m_x = len(list_x)
    n_x = len(list_x[0])
    res_list = []
    for i in range(m_x):
        temp_list = []
        for j in range(n_x):
            temp_list.append(list_x[i][j] * a)
        res_list.append(temp_list)
    return res_list


def naive_logistic_regression(list_x=[], list_y=[], list_w=[], bias=0.0, eta=0.0, sample_num=0, iter_num=0):
    for i in range(iter_num):
        # 第一步：前向传播
        Z = naive_matrix_dot(list_w, list_x)
        Z = naive_matrix_add(Z, bias)
        A = naive_sigmoid(Z)
        # 第二步：反向传播
        dZ = naive_matrix_sub(A, list_y)
        X_T = naive_matrix_transpose(X)
        dZ_XT = naive_matrix_dot(dZ, X_T)
        dW = naive_matrix_product(dZ_XT, (1 / sample_num))
        db = (1 / sample_num) * np.sum(dZ)
        # 第三步：梯度下降
        dW_eta = naive_matrix_product(dW, eta)
        list_w = naive_matrix_sub(list_w, dW_eta)
        bias = bias - eta * db
    return A, list_w, bias


def plot_fitted(X, Y, W, M, A, bias):
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
    x = np.linspace(-3, 3, 100).reshape(100, 1)      # 生成一个x坐标的数组
    y = (-bias - W[0, 0] * x) / W[0, 1]              # 根据x坐标计算y值
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

    # 初始化参数
    feature_num = len(X)                # 特征数
    sample_num = len(X[0])              # 样本个数
    W = [[0.1, 0.08]]                   # 特征值权重
    bias = 0                            # 偏差
    eta = 0.01                          # 步长
    iter_num = 5000                     # 迭代次数
    # 将逻辑回归数据结果画图
    A, W, bias = naive_logistic_regression(X, [Y], W, bias, eta, 20, iter_num)
    plot_fitted(np.mat(X), np.mat(Y), np.mat(W), sample_num, np.mat(A), bias)
    pass

