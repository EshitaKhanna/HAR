# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score


def stat_acc_f1(label, results_estimated):
    # label = np.concatenate(label, 0)
    # results_estimated = np.concatenate(results_estimated, 0)
    label_estimated = np.argmax(results_estimated, 1)
    f1 = f1_score(label, label_estimated, average='macro')
    acc = np.sum(label == label_estimated) / label.size
    return acc, f1


def stat_acc_f1_matrix(label, results_estimated):
    label_estimated = np.argmax(results_estimated, 1)
    f1 = f1_score(label, label_estimated, average='macro')
    acc = np.sum(label == label_estimated) / label.size
    matrix = metrics.confusion_matrix(label, label_estimated) #, normalize='true'
    return acc, f1, matrix


def stat_matrix(label, results_estimated):
    label_estimated = np.argmax(results_estimated, 1)
    matrix = metrics.confusion_matrix(label, label_estimated) #, normalize='true'
    return matrix


