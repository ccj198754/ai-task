#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import os
from HelperClass.NeuralNet import *
from mpl_toolkits.mplot3d import Axes3D

file_name = os.getcwd()+'/mlm.csv'


def showResult(reader, neural):
    X, Y = reader.GetWholeTrainSamples()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(X[:,0], X[:,1], Y[:,0])
    x = np.linspace(0,1)
    y = np.linspace(0,1)
    x, y = np.meshgrid(x, y)
    R = np.hstack((x.ravel().reshape(2500, 1), y.ravel().reshape(2500, 1)))
    z = neural.Forward(R)
    z = z.reshape(-50, 50)
    ax.plot_surface(x, y, z, color='r',alpha=0.5)
    plt.show()

if __name__ == '__main__':
    # data
    reader = DataReader(file_name)
    reader.ReadData()
    reader.NormalizeX()
    reader.NormalizeY()
    # net
    hp = HyperParameters(2, 1, eta=0.1, max_epoch=100, batch_size=5, eps=1e-5)
    net = NeuralNet(hp)
    net.train(reader, checkpoint=0.1)
    print("W = ", net.W)
    print("B = ", net.B)
    showResult(reader,net)
    