#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from HelperClass.NeuralNet import *
from HelperClass.DataReader import *
import os

file_name = os.getcwd()+'/iris.csv'


if __name__ == '__main__':
    #data
    reader = DataReader(file_name)
    reader.ReadData()
    reader.NormalizeX()#标准化特征值
    reader.NormalizeY(NetType.MultipleClassifier, base=1)#将标签值转化成独热编码
    reader.GenerateValidationSet()#产生验证集

    n_input = reader.num_feature
    n_hidden = 4#（4个隐藏神经元）
    # n_hidden = 8
    n_output = 3#三输出
    eta, batch_size, max_epoch,eps  = 0.1,5,10000,1e-3
    

    hp = HyperParameters(n_input, n_hidden, n_output,
                             eta, max_epoch, batch_size, eps,
                             NetType.MultipleClassifier, InitialMethod.Xavier)

    net = NeuralNet(hp, "非线性分类")
    net.train(reader, 10)
    net.ShowTrainingHistory()
    print("===权重矩阵===\nwb1.W = ", net.wb1.W,"\nwb1.B = ", net.wb1.B,"\nwb2.W = ", net.wb2.W,"\nwb2.B = ", net.wb2.B)