# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models, train
from config import load_dataset_label_names
from statistic import stat_acc_f1, stat_matrix
from utils import get_device, handle_argv, Preprocess4Normalization, load_data, Preprocess4Rotation \
    , IMUDataset, set_seeds, prepare_dataset, Preprocess4Sample, print_stat, \
    prepare_classification_da_dataset, IMUDADataset, Preprocess4Mask


def train_ar_cen(args, method, label_index=0, label_max=4, training_size=1000,
                       eval=False):
    data, labels = load_data(args)

    label_names, label_num = load_dataset_label_names(args.dataset_cfg, label_index, label_max)

    set_seeds(args.train_cfg.seed)
    classifier_cfg = args.model_cfg

    classifier = models.fetch_classifier(method, classifier_cfg, input=args.encoder_cfg.hidden, output=label_num)
    model = models.CompositeClassifierDA(args.encoder_cfg, classifier=classifier) # , freeze_decoder=True
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.train_cfg.lr, weight_decay=args.train_cfg.lambda2)
    device = get_device(args.gpu)
    trainer = train.Trainer(args.train_cfg, model, optimizer, args.save_path, device, print_model=False)

    criterion_clf = nn.CrossEntropyLoss()
    alpha = args.train_cfg.alpha

    def loss_mix(model, batch):
        loss = 0.0
        inputs_source, mask_inputs_source, masked_pos_source, inputs_selection_source \
            , inputs_target, mask_inputs_target, masked_pos_target, inputs_selection_target, label = batch

        logits, logits_domain_source = model(inputs_source, True, True)
        loss_clf = criterion_clf(logits, label)
        loss += loss_clf

        if alpha != 0.0:
            _, logits_domain_target = model(inputs_target, True, True)

            label_domain = torch.cat([torch.ones(label.size(0)), torch.zeros(label.size(0))]).long().to(label.device)
            loss_domain = criterion_clf(torch.cat([logits_domain_source, logits_domain_target], dim=0), label_domain)
            loss += alpha * loss_domain

        return loss

    def forward_clf(model, batch):
        logits = model(batch[0], False, True)
        if len(batch[-1].size()) > 1:
            return logits, batch[-1][:, 0]
        return logits, batch[-1]

    seq_len = classifier_cfg.seq_len
    if args.remove_data_augmentation > 0:
        pipeline = [Preprocess4Normalization(args.model_cfg.feature_num), Preprocess4Mask(args.mask_cfg, full_sequence=True)]
    else:
        pipeline = [Preprocess4Normalization(args.model_cfg.feature_num), Preprocess4Sample(seq_len, temporal=0.4)
            , Preprocess4Rotation(), Preprocess4Mask(args.mask_cfg, full_sequence=True)]

    if not eval:
        if args.remove_data_augmentation > 0:
            data_source, label_source, data_target, label_target, data_vali, label_vali, _, _ \
                = prepare_classification_da_dataset(data, labels, seq_len, label_index, training_size=training_size
                                                    , source_domain=args.source_domain, target_domain=args.target_domain
                                                    , label_max=label_max, seed=args.train_cfg.seed,
                                                    domain_num=args.dataset_cfg.domain_label_size)
        else:
            data_source, label_source, data_target, label_target, data_vali, label_vali, _, _ \
                = prepare_classification_da_dataset(data, labels, 0, label_index, training_size=training_size
                                                    , source_domain=args.source_domain, target_domain=args.target_domain
                                                    , label_max=label_max, seed=args.train_cfg.seed, domain_num=args.dataset_cfg.domain_label_size)

        _, _, _, _, data_test, label_test = prepare_dataset(data, labels, seq_len, label_index=label_index
                        , domain=args.target_domain, label_max=label_max, seed=args.train_cfg.seed, merge=True)

        data_set_train = IMUDADataset(data_source, data_target, label_source, pipeline_source=pipeline, pipeline_target=pipeline)
        data_set_vali = IMUDataset(data_vali, label_vali, pipeline=pipeline[:2])
        data_set_test = IMUDataset(data_test, label_test, pipeline=pipeline[:1])
        data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=args.train_cfg.batch_size)
        data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=args.train_cfg.batch_size)
        data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=args.train_cfg.batch_size)
        if args.classifier is None:
            trainer.train(loss_mix, forward_clf, stat_acc_f1, data_loader_train, data_loader_test, data_loader_vali
                          , model_file=args.encoder, load_self=True, print_time=args.print_time>0)
        else:
            trainer.train(loss_mix, forward_clf, stat_acc_f1, data_loader_train, data_loader_test, data_loader_vali
                          , model_file=args.classifier, load_self=True, print_time=args.print_time>0)

    _, _, _, _, data_test_set, label_test_set \
        = prepare_dataset(data, labels, seq_len, label_index=label_index, domain=args.target_domain
                          , label_max=label_max, seed=args.train_cfg.seed, merge=False)
    stat_re = []
    for i in range(len(data_test_set)):
        data_set_eval = IMUDataset(data_test_set[i], label_test_set[i], pipeline=pipeline[:1])
        data_loader_eval = DataLoader(data_set_eval, shuffle=False, batch_size=args.train_cfg.batch_size)
        stat_re.append(trainer.run(forward_clf, data_loader_eval, evaluate=stat_acc_f1, model_file=args.classifier)) #
    stat_re = np.array(stat_re)
    print_stat(stat_re, labels, args.source_domain, args.dataset_cfg.domain_label_size)
    return stat_re


if __name__ == "__main__":
    args = handle_argv('activity_recognition_cen', 'train_supervised.json')
    stat_re = train_ar_cen(args, args.model, eval=args.eval, training_size=args.label_number)


