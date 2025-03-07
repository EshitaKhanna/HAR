# !/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import random

import numpy as np
import torch

from scipy import signal, interpolate
from scipy.stats import special_ortho_group
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from config import create_io_config, load_dataset_config, TrainConfig, MaskConfig, load_model_config


def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu, print_info=True):
    "get device (CPU or GPU)"
    if gpu is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if print_info:
        print("%s (%d GPUs)" % (device, n_gpu))
    return device


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def narray_to_tensor(narray_list):
    for i in range(len(narray_list)):
        item = narray_list[i]
        if item.dtype == np.float32 or item.dtype == np.float64:
            narray_list[i] = torch.from_numpy(item).float()
        else:
            narray_list[i] = torch.from_numpy(item).long()
    return narray_list


def span_mask(seq_len, max_gram=3, p=0.2, goal_num_predict=15):
    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = p * np.power(1 - p, np.arange(max_gram))
    # alpha = 6
    # pvals = np.power(alpha, ngrams) * np.exp(-alpha) / factorial(ngrams)# possion
    pvals /= pvals.sum(keepdims=True)
    mask_pos = set()
    while len(mask_pos) < goal_num_predict:
        n = np.random.choice(ngrams, p=pvals)
        n = min(n, goal_num_predict - len(mask_pos))
        anchor = np.random.randint(seq_len)
        if anchor in mask_pos:
            continue
        for i in range(anchor, min(anchor + n, seq_len - 1)):
            mask_pos.add(i)
    return list(mask_pos)


def filter_dataset(data, label, filter_index=[0, 1]):
    index = np.zeros(data.shape[0], dtype=bool)
    label_new = []
    for i in range(label.shape[0]):
        temp_label = np.unique(label[i, :, filter_index], axis=1)
        if temp_label.shape == (2, 1):
            index[i] = True
            # label_new.append(label[i, 0, :])
            label_new.append(label[i, :, :])
    # print('Before Merge: %d, After Merge: %d' % (data.shape[0], np.sum(index)))
    return data[index], np.array(label_new)


def reshape_data(data_set, seq_len):
    if seq_len == 0:
        return data_set
    result = []
    for item in data_set:
        result.append(item.reshape(item.shape[0] * item.shape[1] // seq_len, seq_len, item.shape[2]))
    return result


def reshape_label(label_set, seq_len_original, seq_len):
    if seq_len == 0:
        return label_set
    result = []
    for item in label_set:
        item = np.repeat(item, seq_len_original // seq_len, axis=0)
        result.append(item)
    return result


def shuffle_data_label(data, label):
    index = np.random.permutation(np.arange(data.shape[0]))
    return data[index, ...], label[index, ...]


def select_random_data_label(data, label, sample_num):
    index = np.random.choice(label.shape[0], sample_num, replace=False)
    return data[index, ...], label[index, ...]


def filter_domain(data, labels, domain, label_domain_index=2):
    index = labels[:, 0, label_domain_index] == (domain - 1)
    return data[index, ...], labels[index, ...]


def filter_domain_list(data, labels, domain_list, label_domain_index=2):
    if (isinstance(domain_list, int) and domain_list == 0) or (isinstance(domain_list, list) and domain_list == [0]):
        return data, labels
    else:
        if isinstance(domain_list, int):
            domain_list = [domain_list] # convert to a list

        #index = np.isin(labels[:, 0, label_domain_index], np.array(domain_list) - 1)
        #index = np.isin(labels[:, 0, label_domain_index], np.array(domain_list))
        index = np.isin(labels[:, 0, label_domain_index], np.array([4]))
        return data[index, ...], labels[index, ...]


def filter_domain_user_list(labels, source_domain, target_domain, label_user_index=1, label_domain_index=2, domain_num=4):
    if source_domain == 0:
        domain_other = [i + 1 for i in range(domain_num)]
    elif target_domain == 0:
        domain_other = [i + 1 for i in range(domain_num)]
        domain_other.remove(source_domain)
    else:
        domain_other = [target_domain]
    user_domain_mapping = np.unique(labels[:, 0, [label_user_index, label_domain_index]], axis=0)
    index = np.isin(user_domain_mapping[:, 1], np.array(domain_other) - 1)
    return user_domain_mapping[index, 0].astype(np.int32)
    # return np.arange(user_domain_mapping.shape[0])[index].astype(np.int32)


def filter_user(data, labels, labels_user, user):
    index = labels_user == user
    return data[index, ...], labels[index, ...]


def filter_label(data, labels, label_target):
    index = labels == label_target
    return data[index, ...], labels[index, ...]


def filter_label_max(data, labels, label_max, label_index=0):
    label_target = np.arange(label_max)

    # shape (num_samples, 1, 3)
    if labels.ndim == 1:
        print("Warning: Reshaping to (num_samples, 1, 3)")
        labels = labels.reshape(-1, 1, 3)

    print("Labels shape in filter_label_max function:", labels.shape)

    #index = np.isin(labels[:, label_index], label_target)
    index = np.isin(labels[:, 0, label_index], label_target)
    if index.ndim > 1:
        index = index.reshape(-1)
    print("Index shape in filter_label_max function:", index.shape)
    return data[index, ...], labels[index, ...]


def filter_act_user_domain(data, labels, act=None, user=None, domain=None, label_act_max=None):
    index = np.ones(labels.shape[0]).astype(np.bool_)
    if act is not None:
        index = index & (labels[:, 0, 0] == act)
    if label_act_max is not None:
        index = index & (labels[:, 0, 0] < label_act_max)
    if user is not None:
        index = index & (labels[:, 0, 1] == user)
    if domain is not None:
        index = index & (labels[:, 0, 2] == (domain - 1))
    return data[index, ...], labels[index, ...]


def check_domain_user_num(labels, domain, label_user_index=1, label_domain_index=2, domain_num=4):
    if domain == 0:
        domain_list = [i + 1 for i in range(domain_num)]
    else:
        domain_list = [domain]
    index = np.isin(labels[:, 0, label_domain_index], np.array(domain_list) - 1)
    return np.unique(labels[index, 0, label_user_index]).size


def separate_user(data, labels, user_label_index=1):
    if labels.size == 0:
        print("Warning: Labels array is empty. Returning empty dataset.")
        return []
    
    #users = np.unique(labels[:, user_label_index])
    users = np.unique(labels[:, 0, user_label_index])
    result = []
    for i in range(users.size):
        # result.append(filter_user(data, labels, labels[:, user_label_index], users[i]))
        result.append(filter_user(data, labels, labels[:, 0, user_label_index], users[i]))
    return result


def separate_dataset(data, labels, domain, label_index, label_max=None):
    print("Labels shape in separate_dataset:", labels.shape)

    if domain is not None:
        # data, labels = filter_domain(data, labels, domain)
        data, labels = filter_domain_list(data, labels, domain)
        print("Labels shape after filter_domain_list:", labels.shape)

    data, labels = filter_dataset(data, labels)
    print("Labels shape after filter_dataset:", labels.shape)
    if label_max is not None and label_max > 0:
        data, labels = filter_label_max(data, labels, label_max)
        print("Labels shape after filter_label_max:", labels.shape)
    dataset = separate_user(data, labels)
    data_set = [item[0] for item in dataset]
    label_set = [item[1][:, label_index] for item in dataset]
    return data_set, label_set


def prepare_dataset(data, labels, seq_len, training_size=0.8, test_size=0.1, domain=0, label_index=0, merge=False, label_max=None, seed=None):
    print("Labels shape in prepare_dataset:", labels.shape)
   
    data_set, label_set = separate_dataset(data, labels, domain, label_index, label_max=label_max)
    result = [[], [], [], [], [], []]
    for i in range(len(data_set)):
        data_train_temp, data_test, label_train_temp, label_test \
            = train_test_split(data_set[i], label_set[i], train_size=training_size, random_state=seed)
        training_size_new = training_size / (1 - test_size)
        data_train, data_vali, label_train, label_vali \
            = train_test_split(data_train_temp, label_train_temp, train_size=training_size_new, random_state=seed)

        [data_train, data_vali, data_test] = reshape_data([data_train, data_vali, data_test], seq_len)
        [label_train, label_vali, label_test] = reshape_label([label_train, label_vali, label_test],
                                                                    data.shape[1], seq_len)
        result[0].append(data_train)
        result[1].append(label_train)
        result[2].append(data_vali)
        result[3].append(label_vali)
        result[4].append(data_test)
        result[5].append(label_test)
    if merge:
        for i in range(len(result)):
            if i % 2 == 0:
                result[i] = np.vstack(result[i])
            else:
                if isinstance(label_index, list):
                    result[i] = np.vstack(result[i])
                else:
                    result[i] = np.concatenate(result[i])

    # save the split data to .npy files
    np.save('x_train.npy', result[0])  # training data
    np.save('y_train.npy', result[1])  # training labels
    np.save('x_vali.npy', result[2])   # validation data
    np.save('y_vali.npy', result[3])   # validation labels
    np.save('x_test.npy', result[4])   # test data
    np.save('y_test.npy', result[5])   # test labels
    print("[INFO] Internally splitted data saved as .npy files")

    return result

    # # load external .npy files
    # x_train = np.load('x_train.npy').astype(np.float32)
    # y_train = np.load('y_train.npy').astype(np.float32)
    # x_vali = np.load('x_vali.npy').astype(np.float32)
    # y_vali = np.load('y_vali.npy').astype(np.float32)
    # x_test = np.load('x_test.npy').astype(np.float32)
    # y_test = np.load('y_test.npy').astype(np.float32)

    # return x_train, y_train, x_vali, y_vali, x_test, y_test


def prepare_classification_dataset(data, labels, seq_len, label_index, training_size, domain=None, label_max=None, seed=None):
    print("Labels shape in prepare_classification_dataset:", labels.shape)
    print("Data shape in prepare_classification_dataset:", data.shape) 
    
    data_train, label_train, data_vali, label_vali, data_test, label_test \
        = prepare_dataset(data, labels, seq_len=0, domain=domain, label_index=label_index, merge=True
                          , label_max=label_max, seed=seed)
    if training_size == 1.0:
        data_train_label, label_train_label = data_train, label_train
    else:
        data_train_label, _, label_train_label, _ = train_test_split(data_train, label_train, train_size=training_size, random_state=seed)
    [data_train_label, data_vali, data_test] = reshape_data([data_train_label, data_vali, data_test], seq_len)
    [label_train_label, label_vali, label_test] = reshape_label([label_train_label, label_vali, label_test], data.shape[1],
                                                          seq_len)
    return data_train_label, label_train_label, data_vali, label_vali, data_test, label_test


def prepare_classification_da_dataset(data, labels, seq_len, label_index, training_size, source_domain=None, target_domain=None
                                      , label_max=None, seed=None, domain_num=4):
    data_train_label, label_train_label, data_vali, label_vali, data_test, label_test \
        = prepare_classification_dataset(data, labels, seq_len, label_index, training_size
                                         , domain=source_domain, label_max=label_max, seed=seed)
    if target_domain == 0:
        domain_other = [i + 1 for i in range(domain_num)]
        domain_other.remove(source_domain)
    else:
        domain_other = [target_domain]
    data_train_target, label_train_target, _, _, _, _ \
        = prepare_classification_dataset(data, labels, seq_len, label_index, 1.0
                                         , domain=domain_other, label_max=label_max, seed=seed)
    return data_train_label, label_train_label, data_train_target, label_train_target, data_vali, label_vali, data_test, label_test


def select_by_metric(label, select_num, metric=None, random_rate=0.0):
    select_index = np.zeros(label.size, dtype=np.bool)
    label_count = [0] * np.unique(label).size
    if random_rate < 1.0:
        loss_sort_index = metric.argsort()
        top_k = int(select_num * (1 - random_rate))
        for i in range(metric.size):
            if label_count[int(label[loss_sort_index[i]])] < top_k:
                select_index[loss_sort_index[i]] = True
                label_count[int(label[loss_sort_index[i]])] += 1
                if label_count == [top_k] * len(label_count):
                    break
    if random_rate > 0.0:
        index_random = np.random.permutation(label.size)
        for i in range(label.size):
            if label_count[int(label[index_random[i]])] < select_num and not select_index[index_random[i]]:
                select_index[index_random[i]] = True
                label_count[int(label[index_random[i]])] += 1
                if label_count == [select_num] * len(label_count):
                    break
    return select_index


class Pipeline:
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Preprocess4Normalization(Pipeline):

    def __init__(self, feature_len, norm_acc=True, norm_mag=True, gamma=1.0):
        super().__init__()
        self.feature_len = feature_len
        self.norm_acc = norm_acc
        self.norm_mag = norm_mag
        self.eps = 1e-5
        self.acc_norm = 9.8
        self.gamma = gamma

    def __call__(self, instance):
        instance_new = instance.copy()[:, :self.feature_len]
        if instance_new.shape[1] >= 6 and self.norm_acc:
            instance_new[:, :3] = instance_new[:, :3] / self.acc_norm
        if instance_new.shape[1] == 9 and self.norm_mag:
            mag_norms = np.linalg.norm(instance_new[:, 6:9], axis=1) + self.eps
            mag_norms = np.repeat(mag_norms.reshape(mag_norms.size, 1), 3, axis=1)
            instance_new[:, 6:9] = instance_new[:, 6:9] / mag_norms * self.gamma
        return instance_new


class Preprocess4Sample(Pipeline):

    def __init__(self, seq_len, temporal=0.4, temporal_range=[0.8, 1.2]):
        super().__init__()
        self.seq_len = seq_len
        self.temporal = temporal
        self.temporal_range = temporal_range

    def __call__(self, instance):
        if instance.shape[0] == self.seq_len:
            return instance
        if self.temporal > 0:
            temporal_prob = np.random.random()
            if temporal_prob < self.temporal:
                x = np.arange(instance.shape[0])
                ratio_random = np.random.random() * (self.temporal_range[1] - self.temporal_range[0]) + self.temporal_range[0]
                seq_len_scale = int(np.round(ratio_random * self.seq_len))
                index_rand = np.random.randint(0, high=instance.shape[0] - seq_len_scale)
                instance_new = np.zeros((self.seq_len, instance.shape[1]))
                for i in range(instance.shape[1]):
                    f = interpolate.interp1d(x, instance[:, i], kind='linear')
                    x_new = index_rand + np.linspace(0, seq_len_scale, self.seq_len)
                    instance_new[:, i] = f(x_new)
                return instance_new
        index_rand = np.random.randint(0, high=instance.shape[0] - self.seq_len)
        return instance[index_rand:index_rand + self.seq_len, :]


class Preprocess4Noise(Pipeline):

    def __init__(self, p=1.0, mu=0.0, var=0.1):
        super().__init__()
        self.p = p
        self.mu = mu
        self.var = var

    def __call__(self, instance):
        if np.random.random() < self.p:
            instance += np.random.normal(self.mu, self.var, instance.shape)
        return instance


class Preprocess4Rotation(Pipeline):

    def __init__(self, sensor_dimen=3):
        super().__init__()
        self.sensor_dimen = sensor_dimen

    def __call__(self, instance):
        return self.rotate_random(instance)

    def rotate_random(self, instance):
        # print("Original instance shape:", instance.shape)
        instance_new = instance.reshape(instance.shape[0], instance.shape[1] // self.sensor_dimen, self.sensor_dimen)
        rotation_matrix = special_ortho_group.rvs(self.sensor_dimen)
        # print("Rotation matrix:\n", rotation_matrix)
        for i in range(instance_new.shape[1]):
            instance_new[:, i, :] = np.dot(instance_new[:, i, :], rotation_matrix)
        # print("Instance shape after rotation and reshaping back:", reshaped_instance.shape)
        return instance_new.reshape(instance.shape[0], instance.shape[1])


class Preprocess4Permute(Pipeline):

    def __init__(self, segment_size=4):
        super().__init__()
        self.segment_size = segment_size

    def __call__(self, instance):
        original_shape = instance.shape
        instance = instance.reshape(self.segment_size, instance.shape[0]//self.segment_size, -1)
        order = np.random.permutation(self.segment_size)
        instance = instance[order, :, :]
        return instance.reshape(original_shape)


class Preprocess4Flip(Pipeline):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, instance):
        if np.random.random() < self.p:
            instance = instance[::-1, :].copy()
        return instance


class Preprocess4STFT(Pipeline):

    def __init__(self, window=50, cut_off_frequency=17, fs=20):
        super().__init__()
        self.window = window
        self.cut_off_frequency = cut_off_frequency
        self.fs = fs

    def __call__(self, instance):
        instance_new = []
        for i in range(instance.shape[1]):
            f, t, Zxx = signal.stft(instance[:, i], self.fs, nperseg=self.window)
            instance_new.append(Zxx[:self.cut_off_frequency, :])
        instance_new = np.abs(np.vstack(instance_new).transpose([1, 0]))
        return instance_new


class Preprocess4Mask:

    def __init__(self, mask_cfg, full_sequence=False):
        self.mask_ratio = mask_cfg.mask_ratio  # masking probability
        self.mask_alpha = mask_cfg.mask_alpha
        self.max_gram = mask_cfg.max_gram
        self.mask_prob = mask_cfg.mask_prob
        self.replace_prob = mask_cfg.replace_prob
        self.full_sequence = full_sequence

    def gather(self, data, position1, position2):
        result = []
        for i in range(position1.shape[0]):
            result.append(data[position1[i], position2[i]])
        return np.array(result)

    def mask(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.zeros(position2[i].size)
        return data

    def replace(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.random.random(position2[i].size)
        return data

    def __call__(self, instance):
        shape = instance.shape

        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = max(1, int(round(shape[0] * self.mask_ratio)))

        # For masked Language Models
        # mask_pos = bert_mask(shape[0], n_pred)
        mask_pos = span_mask(shape[0], self.max_gram,  goal_num_predict=n_pred)

        instance_mask = instance.copy()
        if np.random.rand() < self.mask_prob:
            instance_mask[mask_pos, :] = np.zeros((len(mask_pos), shape[1]))
        elif np.random.rand() < self.mask_prob + self.replace_prob:
            instance_mask[mask_pos, :] = np.random.random((len(mask_pos), shape[1]))
        seq = instance[mask_pos, :]
        if self.full_sequence:
            return instance, instance_mask, np.array(mask_pos), np.array(seq)
        else:
            return instance_mask, np.array(mask_pos), np.array(seq)


class IMUDataset(Dataset):

    def __init__(self, data, labels=None, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.labels = labels.astype(np.int32) if labels is not None else labels

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        result = []
        if isinstance(instance, tuple):
            result += list(instance)
        else:
            result += [instance]
        if self.labels is not None:
            result.append(np.array(self.labels[index]))
        return narray_to_tensor(list(result))

    def __len__(self):
        return len(self.data)


class IMUDADataset(Dataset):

    def __init__(self, data_source, data_target, labels, pipeline_source=[], pipeline_target=[], mode='cut'):
        super().__init__()
        self.pipeline_source = pipeline_source
        self.pipeline_target = pipeline_target
        self.data_source = data_source
        self.mode = mode
        self.data_target = data_target
        if mode == 'cut':
            index = np.random.choice(data_target.shape[0], data_source.shape[0]
                                     , replace=data_source.shape[0] > data_target.shape[0])
            self.data_target = self.data_target[index]
        self.labels = labels.astype(np.int32) # labels of data1

    def __getitem__(self, index):
        instance1 = self.data_source[index]
        if self.mode == 'cut':
            instance2 = self.data_target[index]
        else:
            instance2 = self.data_target[np.random.randint(self.data_target.shape[0])]
        for proc in self.pipeline_source:
            instance1 = proc(instance1)
        for proc in self.pipeline_target:
            instance2 = proc(instance2)
        result = []
        if isinstance(instance1, tuple):
            result += list(instance1)
        else:
            result += [instance1]
        if isinstance(instance2, tuple):
            result += list(instance2)
        else:
            result += [instance2]
        result.append(np.array(self.labels[index]))
        return narray_to_tensor(list(result))

    def __len__(self):
        return len(self.data_source)


class FFTDataset(Dataset):
    def __init__(self, data, labels, mode=0, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.labels = labels
        self.mode = mode

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        seq = self.preprocess(instance)
        return torch.from_numpy(seq), torch.from_numpy(np.array(self.labels[index])).long()

    def __len__(self):
        return len(self.data)

    def preprocess(self, instance):
        f = np.fft.fft(instance, axis=0, n=10)
        mag = np.abs(f)
        phase = np.angle(f)
        return np.concatenate([mag, phase], axis=0).astype(np.float32)


def handle_argv(target, config_train):
    parser = argparse.ArgumentParser(description='UniHAR Framework')
    parser.add_argument('model', type=str, help='Model.')
    parser.add_argument('dataset', type=str, help='Dataset name.')
    parser.add_argument('dataset_version', type=str, help='Dataset version')
    parser.add_argument('-d', '--domain', type=int, default=0, help='The domain index.')
    parser.add_argument('-sd', '--source_domain', type=int, default=0, help='The source domain index.')
    parser.add_argument('-td', '--target_domain', type=int, default=0, help='The target domain index')
    parser.add_argument('-e', '--encoder', type=str, default=None, help='Pretrain encoder and decoder file.')
    parser.add_argument('-c', '--classifier', type=str, default=None, help='Trained classifier file.')
    parser.add_argument('-g', '--gpu', type=str, default=None, help='Set specific GPU.')
    parser.add_argument('-t', '--cfg_train', type=str, default='./config/' + config_train, help='Training config json file path')
    parser.add_argument('-m', '--cfg_mask', type=str, default='./config/mask.json', help='Mask strategy json file path')
    parser.add_argument('-l', '--label_index', type=int, default=0, help='Label Index')
    parser.add_argument('-n', '--label_number', type=int, default=1000, help='Label number')
    parser.add_argument('-rda', '--remove_data_augmentation', type=int, default=0, help='Specify whether adopting data augmentations.')
    parser.add_argument('-s', '--save_name', type=str, default='model', help='The saved model name')
    parser.add_argument('-pt', '--print_time', type=int, default=0, help='Label number')
    parser.add_argument('--eval', action='store_true', help='Specify evaluation mode.')
    args = parser.parse_args()
    args = load_model_config(args, target, args.model)
    args.dataset_cfg = load_dataset_config(args.dataset, args.dataset_version)
    args = create_io_config(args, target=target)
    args.train_cfg = TrainConfig.from_json(args.cfg_train)
    args.mask_cfg = MaskConfig.from_json(args.cfg_mask)
    return args


def manual_models_path(args, target, models):
    save_path_pretrain = os.path.join('saved', target + "_" + args.dataset + "_" + args.dataset_version)
    models_new = [os.path.join(save_path_pretrain, model) for model in models]
    return models_new


def read_dataset(path, version):
    path_data = os.path.join(path, 'data_' + version + '.npy')
    path_label = os.path.join(path, 'label_' + version + '.npy')
    return np.load(path_data), np.load(path_label)


def load_data(args):
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels


def load_classifier_data_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_cfg, dataset_cfg


def load_classifier_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    set_seeds(train_cfg.seed)
    return train_cfg, model_cfg, dataset_cfg


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_stat(stats, labels, source_domain, domain_size):
    print_info = ['All domain users', 'Other domain users', 'Cross domain users']
    message = ''
    for i in range(domain_size + 2):
        sd = 0 if i == 0 else source_domain
        td = i - 1 if i > 1 else 0
        user_index = filter_domain_user_list(labels, sd, td, domain_num=domain_size)
        mean_acc_f1 = np.mean(stats[user_index, :], axis=0)
        std_acc_f1 = np.std(stats[user_index, :], axis=0)
        if i > 1:
            message += '[%0.3f, %0.3f; %0.3f, %0.3f]' % (mean_acc_f1[0], std_acc_f1[0], mean_acc_f1[1], std_acc_f1[1])
        else:
            print(print_info[i])
            print('[Accuracy; F1]: [%0.3f, %0.3f; %0.3f, %0.3f]' % (mean_acc_f1[0], std_acc_f1[0], mean_acc_f1[1], std_acc_f1[1]))
        if i == domain_size + 1:
            pass
            # print(print_info[-1])
            # print(message)

