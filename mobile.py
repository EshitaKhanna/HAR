# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

import models
from config import load_dataset_label_names
from models import fetch_classifier
from utils import handle_argv, get_device, load_classifier_config, \
    count_model_parameters


def output_classifier_model(args, method, label_index=0):

    label_names, label_num = load_dataset_label_names(args.dataset_cfg, label_index, label_max=4)
    classifier_cfg = args.model_cfg

    classifier = models.fetch_classifier(method, classifier_cfg, input=args.encoder_cfg.hidden, output=label_num)
    model = models.CompositeMobileClassifier(args.encoder_cfg, classifier=classifier)
    model.eval()
    print("Number of parameters: %d" % count_model_parameters(model))
    example = torch.rand(1, classifier_cfg.seq_len, classifier_cfg.feature_num)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module_optimized = optimize_for_mobile(traced_script_module)
    traced_script_module_optimized._save_for_lite_interpreter(args.classifier + "_mobile.pt")


if __name__ == "__main__":
    args = handle_argv('activity_recognition_decen', 'train_supervised.json')
    output_classifier_model(args, args.model)

