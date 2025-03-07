# !/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import sys
from typing import NamedTuple
import os
# from bunch import bunchify


class PretrainModelConfig(NamedTuple):

    feature_num: int = 0  #
    hidden: int = 0  # Dimension of Hidden Layer in Transformer Encoder
    hidden_ff: int = 0  # Dimension of Intermediate Layers in Positionwise Feedforward Net
    n_layers: int = 0  # Numher of Hidden Layers
    n_heads: int = 0  # Numher of Heads in Multi-Headed Attention Layers
    seq_len: int = 0  # Maximum Length for Positional Embeddings
    emb_norm: bool = False

    @classmethod
    def from_json(cls, js):
        return cls(**js)


class ClassifierModelConfig(NamedTuple):
    "Configuration for classifier model"
    seq_len: int = 0
    feature_num: int = 0

    num_rnn: int = 0
    num_layers: int = 0
    rnn_io: list = []
    rnn_bidirection: list = []

    num_cnn: int = 0
    conv_io: list = []
    pool: list = []
    flat_num: int = 0

    num_attn: int = 0
    num_head: int = 0
    atten_hidden: int = 0

    num_linear: int = 0
    linear_io: list = []

    activ: bool = False
    dropout: bool = False

    encoder: str = None
    encoder_cfg: object = PretrainModelConfig


    @classmethod
    def from_json(cls, js):
        return cls(**js)

    @classmethod
    def from_encoder_cfg(cls, js, encoder_cfg):

        cfg = cls(**js)
        cfg.encoder_cfg = encoder_cfg
        return cfg


class TrainConfig(NamedTuple):
    """ Hyperparameters for training """
    seed: int = 0  # random seed
    batch_size: int = 0
    lr: int = 0  # learning rate
    n_epochs: int = 0  # the number of epoch
    num_round: int = 0  # the number of rounds in federated learning
    warmup: float = 0
    save_steps: int = 0  # interval for saving model
    total_steps: int = 0  # total number of steps to train
    lambda1: float = 0
    lambda2: float = 0
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


class MaskConfig(NamedTuple):
    """ Hyperparameters for training """
    mask_ratio: float = 0  # masking probability
    mask_alpha: int = 0  # How many tokens to form a group.
    max_gram: int = 0  # number of max n-gram to masking
    mask_prob: float = 1.0
    replace_prob: float = 0.0

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))


class DatasetConfig(NamedTuple):
    # dataset = Narray with shape (size, seq_len, dimension)
    sr: int = 0  # sampling rate
    size: int = 0  # data sample number
    seq_len: int = 0  # seq length
    dimension: int = 0  # feature dimension

    activity_label_index: int = -1  # index of activity label
    activity_label_size: int = 0  # number of activity label
    activity_label: list = []  # names of activity label.

    user_label_index: int = -1  # index of user label
    user_label_size: int = 0  # number of user label

    position_label_index: int = -1  # index of phone position label
    position_label_size: int = 0  # number of position label
    position_label: list = []  # names of position label.

    model_label_index: int = -1  # index of phone model label
    model_label_size: int = 0  # number of model label

    domain_label_index: int = -1  # index of phone model label
    domain_label_size: int = 0  # number of model label

    note:str = ""

    @classmethod
    def from_json(cls, js):
        return cls(**js)


def create_io_config(args, target):
    data_path = os.path.join('dataset', args.dataset, 'data_' + args.dataset_version + '.npy')
    label_path = os.path.join('dataset', args.dataset, 'label_' + args.dataset_version + '.npy')
    args.data_path = data_path
    args.label_path = label_path

    save_folder = os.path.join('saved', target + "_" + args.dataset + "_" + args.dataset_version)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    args.save_path = os.path.join(save_folder, args.save_name)

    if args.classifier is not None:
        args.classifier = os.path.join(save_folder, args.classifier)

    if args.encoder is not None:
        save_path_encoder = os.path.join('saved', "feature_extraction_" + args.dataset + "_" + args.dataset_version)
        args.encoder = os.path.join(save_path_encoder, args.encoder)
    return args


def load_model_config(args, target, model, path_bert='config/encoder.json', path_classifier='config/classifier.json'):
    model_cfg = None
    if "fe" in target:
        model_config_all = json.load(open(path_bert, "r"))
        if model in model_config_all:
            model_cfg = PretrainModelConfig.from_json(model_config_all[model])
    else:
        model_config_all = json.load(open(path_classifier, "r"))
        if model in model_config_all:
            model_cfg = ClassifierModelConfig.from_json(model_config_all[model])
            if model_cfg.encoder is not None:
                model_encoder_config_all = json.load(open(path_bert, "r"))
                if model_cfg.encoder in model_encoder_config_all:
                    encoder_cfg = PretrainModelConfig.from_json(model_encoder_config_all[model_cfg.encoder])
                    args.encoder_cfg = encoder_cfg
                else:
                    print("Unable to find model encoder config!")
    if model_cfg is None:
        print("Unable to find model config!")
        sys.exit()
    args.model_cfg = model_cfg
    return args


def load_dataset_config(dataset, version):
    path = 'dataset/data_config.json'
    dataset_config_all = json.load(open(path, "r"))
    name = dataset + "_" + version
    if name in dataset_config_all:
        return DatasetConfig.from_json(dataset_config_all[name])
    print("Unable to find dataset config!")
    sys.exit()


def load_dataset_label_names(dataset_config, label_index=0, label_max=0):
    for p in dir(dataset_config):
        if getattr(dataset_config, p) == label_index and "label_index" in p:
            temp = p.split("_")
            label_num = getattr(dataset_config, temp[0] + "_" + temp[1] + "_size")
            if hasattr(dataset_config, temp[0] + "_" + temp[1]):
                if label_max == 0:
                    return getattr(dataset_config, temp[0] + "_" + temp[1]), label_num
                else:
                    return getattr(dataset_config, temp[0] + "_" + temp[1])[:label_max], label_max
            else:
                return None, label_num
    return None, -1

