# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import models, train
from config import load_dataset_config, load_dataset_label_names, load_model_config
from loss import MMDLoss
from statistic import stat_acc_f1, stat_matrix
from utils import get_device, handle_argv, Preprocess4Normalization, load_data, Preprocess4Rotation \
    , IMUDataset, \
    set_seeds, prepare_classification_dataset, prepare_dataset, Preprocess4Sample, print_stat, Preprocess4Mask, \
    Preprocess4Noise, Preprocess4Permute


def train_ar_decen(args, method, label_index=0, label_max=4, training_size=1000,
                   eval=False):
    data, labels = load_data(args)
    label_names, label_num = load_dataset_label_names(args.dataset_cfg, label_index, label_max)

    set_seeds(args.train_cfg.seed)
    classifier_cfg = args.model_cfg

    classifier = models.fetch_classifier(method, classifier_cfg, input=args.encoder_cfg.hidden, output=label_num)
    model = models.CompositeClassifier(args.encoder_cfg, classifier=classifier) # , freeze_decoder=True
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.train_cfg.lr, weight_decay=args.train_cfg.lambda2)
    device = get_device(args.gpu)
    trainer = train.Trainer(args.train_cfg, model, optimizer, args.save_path, device, print_model=False)

    criterion = nn.CrossEntropyLoss()

    def loss_clf(model, batch):
        inputs, mask_inputs, masked_pos, inputs_selection, label = batch

        if label.dim() > 1:
            label = label[:, 0].long()  

        logits = model(inputs, training=True)
        loss = criterion(logits, label)
        return loss

    def forward_clf(model, batch):
        logits = model(batch[0], training=False)
        if len(batch[-1].size()) > 1:
            return logits, batch[-1][:, 0]
        return logits, batch[-1]

    seq_len = classifier_cfg.seq_len
    
    # pipeline (data augmentation)
    if args.remove_data_augmentation == 1:
        pipeline = [Preprocess4Normalization(args.model_cfg.feature_num),
                    Preprocess4Mask(args.mask_cfg, full_sequence=True)]
    elif args.remove_data_augmentation == 2:
        pipeline = [Preprocess4Normalization(args.model_cfg.feature_num), Preprocess4Sample(seq_len, temporal=0.0)
            , Preprocess4Rotation(), Preprocess4Mask(args.mask_cfg, full_sequence=True)]
    elif args.remove_data_augmentation == 3:
        pipeline = [Preprocess4Normalization(args.model_cfg.feature_num), Preprocess4Sample(seq_len, temporal=0.0)
            , Preprocess4Rotation(), Preprocess4Noise(), Preprocess4Permute(),
                    Preprocess4Mask(args.mask_cfg, full_sequence=True)]
    else:
        pipeline = [Preprocess4Normalization(args.model_cfg.feature_num), Preprocess4Sample(seq_len, temporal=0.4)
            , Preprocess4Rotation(), Preprocess4Mask(args.mask_cfg, full_sequence=True)]

    if not eval:
        if args.remove_data_augmentation == 1:
            data_train, label_train, data_vali, label_vali, _, _ \
                = prepare_classification_dataset(data, labels, seq_len, label_index, training_size=training_size
                                                 , domain=args.target_domain, label_max=label_max,
                                                 seed=args.train_cfg.seed)
        else:
            data_train, label_train, data_vali, label_vali, _, _ \
                = prepare_classification_dataset(data, labels, 0, label_index,  training_size=training_size
                                           , domain=args.target_domain, label_max=label_max, seed=args.train_cfg.seed)

        # Prepare the datasets and data loaders
        _, _, _, _, data_test, label_test = prepare_dataset(data, labels, seq_len, label_index=label_index
                        , domain=args.target_domain, label_max=label_max, seed=args.train_cfg.seed, merge=True)

        data_set_train = IMUDataset(data_train, label_train, pipeline=pipeline)
        data_set_vali = IMUDataset(data_vali, label_vali, pipeline=pipeline[:2])
        data_set_test = IMUDataset(data_test, label_test, pipeline=pipeline[:1])
        data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=args.train_cfg.batch_size)
        data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=args.train_cfg.batch_size)
        data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=args.train_cfg.batch_size)

        # Perform training
        trainer.train(loss_clf, forward_clf, stat_acc_f1, data_loader_train, data_loader_test, data_loader_vali
                      , model_file=args.encoder, load_self=True, print_time=args.print_time>0)

    _, _, _, _, data_test_set, label_test_set \
        = prepare_dataset(data, labels, seq_len, label_index=label_index, domain=args.target_domain
                          , label_max=label_max, seed=args.train_cfg.seed, merge=False)
    stat_re = []
    for i in range(len(data_test_set)):
        data_set_eval = IMUDataset(data_test_set[i], label_test_set[i], pipeline=pipeline[:1])
        data_loader_eval = DataLoader(data_set_eval, shuffle=False, batch_size=args.train_cfg.batch_size)
        stat_re.append(trainer.run(forward_clf, data_loader_eval, evaluate=stat_acc_f1, model_file=args.classifier))
    stat_re = np.array(stat_re)
    print_stat(stat_re, labels, args.source_domain, args.dataset_cfg.domain_label_size)
    return stat_re


def pretrain_encoder(args):
    # load data 
    data, _ = load_data(args)
    args.dataset_cfg = load_dataset_config(args.dataset, args.dataset_version)
    args = load_model_config(args, target="feature_extraction", model=args.model)
    encoder_cfg = args.model_cfg 
    
    # initialize pretrain model
    encoder = models.BertModel4Pretrain(encoder_cfg)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.train_cfg.lr)
    device = get_device(args.gpu)
    trainer = train.Trainer(args.train_cfg, encoder, optimizer, args.save_path, device)

    seq_len = args.model_cfg.seq_len

    pipeline = [Preprocess4Normalization(args.model_cfg.feature_num), Preprocess4Sample(seq_len, temporal=0.4)
            , Preprocess4Rotation(), Preprocess4Mask(args.mask_cfg, full_sequence=True)]

    # create dataloaders
    data_train, data_val = train_test_split(data, test_size=0.2, random_state=args.train_cfg.seed)
    train_dataset = IMUDataset(data_train, pipeline=pipeline)
    val_dataset = IMUDataset(data_val, pipeline=pipeline)
    
    train_loader = DataLoader(train_dataset, batch_size=args.train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.train_cfg.batch_size, shuffle=False)

    def loss_fn(model, batch):
        inputs, masked_inputs, masked_pos, original_seqs = batch
        reconstructed = model(masked_inputs, masked_pos)
        return F.mse_loss(reconstructed, original_seqs)
    
    def forward_fn(model, batch):
        inputs, masked_inputs, masked_pos, original_seqs = batch
        output = model(masked_inputs, masked_pos)
        return output, original_seqs

    def eval_fn(true, pred):
        return ((true - pred) ** 2).mean().item()
    
    trainer.pretrain(get_loss=loss_fn,
                    forward=forward_fn,
                    evaluate=eval_fn,
                    data_loader=train_loader,
                    data_loader_test=val_loader,
                    model_file=None,   
                    load_self=False,
                    save=True,
                    num_epochs=args.train_cfg.n_epochs)
    
if __name__ == "__main__":
    if sys.argv[1] == "pretrain":
        sys.argv.pop(1)
        args = handle_argv('feature_extraction', 'train_baseline.json')
        pretrain_encoder(args)
    else:
        args = handle_argv('activity_recognition_decen', 'train_supervised.json')

        args.source_domain = 1  # HHAR - source domain
        args.target_domain = 4  # dummy data - target domain

        args.dataset = "dummy"
        args.dataset_version = "20_120"
        args.label_number = 0.8

        stat_re = train_ar_decen(args, args.model, eval=args.eval, training_size=args.label_number)


