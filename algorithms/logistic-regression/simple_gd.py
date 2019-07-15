#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/9 7:57
# @Author  : ysrs
# @Site    : https://www.zhihu.com/people/ysrs
# @File    : simple_gd.py
# @Software: PyCharm


# 原始函数
def func(x):
    return 4*x*x + 5*x + 1


# 原始函数的导数
def derivative_func(x):
    return 8*x + 5


def gd(iter_num=10, step=0.1, x_init=0):
    x = x_init
    for i in range(iter_num):
        x = x - step * derivative_func(x)
        print('x:' + str(x))


if __name__ == '__main__':
    gd(10, 0.1, 0)

