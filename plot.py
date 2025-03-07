# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

SENSORS = ["Accelerometer", "Gyroscope", "Magnetometer"]
SENSOR_NAMES = ['ACC-X', 'ACC-Y', 'ACC-Z', 'GYRO-X', 'GYRO-Y', 'GYRO-Z', 'MAG-X', 'MAG-Y', 'MAG-Z']
COLOR_BLUE = 'tab:blue'; COLOR_ORANGE = 'tab:orange'; COLOR_GREEN = 'tab:green'
COLOR_RED = 'tab:red'; COLOR_PURPLE = 'tab:purple'; COLOR_BROWN = 'tab:brown'
COLOR_LIST = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_RED, COLOR_PURPLE, COLOR_BROWN]
LINE_STYLES = ['solid', 'dotted']
MAKER_STYLES = ['o', '^', 's', 'P', 'D']
SENSOR_DIMENSION_NAMES = ["X", "Y", "Z"]
LINE_WIDTH = 3.0


def plot_tsne(data, labels, dimension=2, label_names=None, group_num=None):
    tsne = TSNE(n_components=dimension)
    if data.ndim > 2:
        data = data.reshape(data.shape[0], -1)
    data_ = tsne.fit_transform(data)
    ls = np.unique(labels)
    plt.figure(figsize=(10, 6), dpi=80)
    bwith = 2
    TK = plt.gca()
    TK.spines['bottom'].set_linewidth(bwith)
    TK.spines['left'].set_linewidth(bwith)
    TK.spines['top'].set_linewidth(bwith)
    TK.spines['right'].set_linewidth(bwith)
    for i in range(ls.size):
        index = labels == ls[i]
        x = data_[index, 0]
        y = data_[index, 1]
        if label_names is None:
            plt.scatter(x, y, label=str(int(ls[i])))
        else:
            if group_num is None:
                plt.scatter(x, y, label=label_names[int(ls[i])])
            else:
                plt.scatter(x, y, label=label_names[int(ls[i])]
                            , marker=MAKER_STYLES[i // group_num], c=COLOR_LIST[i % group_num])
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='lower right') #, prop={'size': 20, 'weight':'bold'}
    plt.show()
    return data_


def plot_matrix(matrix, labels_name=None):
    plt.figure()
    row_sum = matrix.sum(axis=1)
    matrix_per = np.copy(matrix).astype('float')
    for i in range(row_sum.size):
        if row_sum[i] != 0:
            matrix_per[i] = matrix_per[i] / row_sum[i]
    # plt.figure(figsize=(10, 7))
    if labels_name is None:
        labels_name = "auto"
    sn.heatmap(matrix_per, annot=True, fmt='.2f', xticklabels=labels_name, yticklabels=labels_name)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    # plt.savefig()
    return matrix


def plot_sensor(sensor, sensor_re=None, sensor_dimen=3, plot_magnitude=False, sensor_name="Sensor"):
    plt.figure()
    plt.suptitle(sensor_name) # Sensor Reconstruction Comparison
    x = np.arange(sensor.shape[0])
    plt.xlabel("Index")
    plt.ylabel(sensor_name)
    for i in range(sensor_dimen):
        plt.plot(x, sensor[:, i], label=SENSOR_DIMENSION_NAMES[i], linestyle=LINE_STYLES[0], color=COLOR_LIST[i]
                 , linewidth=LINE_WIDTH)  #
        if sensor_re is not None:
            plt.plot(x, sensor_re[:, i], label=SENSOR_NAMES[i], linestyle=LINE_STYLES[1], color=COLOR_LIST[i]
                     , linewidth=LINE_WIDTH)
    if plot_magnitude:
        plt.plot(x, np.linalg.norm(sensor, axis=1), label="M", linestyle=LINE_STYLES[0], color="tab:gray"
                 , linewidth=LINE_WIDTH)
        if sensor_re is not None:
            plt.plot(x, np.linalg.norm(sensor_re, axis=1), label="M-R", linestyle=LINE_STYLES[1], color="tab:gray"
                     , linewidth=LINE_WIDTH)
    plt.legend()
    plt.show()


