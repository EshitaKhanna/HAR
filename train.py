# !/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import os
import sys
import time
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn

from utils import count_model_parameters


class Trainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, model, optimizer, save_path, device, print_model=True):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.save_path = save_path
        self.device = device
        self.print_model = print_model

    def pretrain(self, get_loss, forward, evaluate, data_loader, model_file=None, data_parallel=False, data_loader_test=None, load_self=False, save=True, num_epochs=None):
        """ Train Loop """
        self.load(model_file, load_self=load_self)
        model = self.model.to(self.device)
        if data_parallel:
            model = nn.DataParallel(model)
        model_best = None
        global_step = 0
        best_loss = 1e6
        n_epochs = num_epochs if num_epochs is not None else self.cfg.n_epochs
        for e in range(n_epochs):
            loss_sum = 0.
            time_sum = 0.0
            self.model.train()
            for i, batch in enumerate(data_loader):
                batch = [t.to(self.device) for t in batch]
                start_time = time.time()
                self.optimizer.zero_grad()
                loss = get_loss(model, batch)
                if torch.sum(torch.isnan(loss)) > 0:
                    print(torch.sum(torch.isnan(loss)))
                    sys.exit()
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                time_sum += time.time() - start_time
                global_step += 1
                loss_sum += loss.item()

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('The Total Steps have been reached.')
                    return

            loss_eva = self.run(forward, data_loader_test, evaluate=evaluate)
            print('Epoch %d/%d : Average Loss %5.4f. Test Loss %5.4f'
                    % (e + 1, self.cfg.n_epochs, loss_sum / len(data_loader), loss_eva))
            # print("Training time: %.5f seconds" % (time_sum / len(data_loader)))
            if loss_eva < best_loss:
                best_loss = loss_eva
                model_best = copy.deepcopy(model.state_dict()) #copy.deepcopy(model)
                if save:
                    self.save()
        self.model.load_state_dict(model_best)
        print('The Total Epoch have been reached.')

    def run(self, forward, data_loader, evaluate=None, model_file=None, data_parallel=False, load_self=False):
        """ Evaluation Loop """
        self.model.eval()
        self.load(model_file, load_self=load_self)
        model = self.model.to(self.device)

        results = [] # prediction results
        labels = []
        time_sum = 0.0
        for batch in data_loader:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad(): # evaluation without gradient calculation
                start_time = time.time()
                result, label = forward(model, batch)
                time_sum += time.time() - start_time
                result = result
                results.append(result)
                if label is not None:
                    labels.append(label)
        # print("Eval execution time: %.5f seconds" % (time_sum / len(dt)))
        if evaluate:
            return evaluate(torch.cat(labels, 0).cpu().numpy(), torch.cat(results, 0).cpu().numpy())
        else:
            return torch.cat(results, 0).cpu().numpy()

    def train(self, get_loss, forwards, evaluate, data_loader_train, data_loader_test, data_loader_vali, model_file=None
              , data_parallel=False, load_self=False, print_time=False):
        """ Train Loop """
        self.load(model_file, load_self)
        model = self.model.to(self.device)

        global_step = 0 # global iteration steps regardless of epochs
        vali_acc_best = 0.0
        best_stat = tuple([0] * 6)
        model_best = None
        for e in range(self.cfg.n_epochs):
            loss_sum = 0.0 # the sum of iteration losses to get average loss in every epoch
            time_sum = 0.0
            self.model.train()
            for i, batch in enumerate(data_loader_train):
                batch = [t.to(self.device) for t in batch]

                start_time = time.time()
                self.optimizer.zero_grad()
                loss = get_loss(model, batch)

                loss = loss.mean() # mean() for Data Parallelism
                loss.backward()
                self.optimizer.step()

                global_step += 1
                loss_sum += loss.item()
                time_sum += time.time() - start_time
                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('The Total Steps have been reached.')
                    return
            if isinstance(forwards, list):
                train_acc, train_f1 = self.run(forwards[0], data_loader_train, evaluate)
                test_acc, test_f1 = self.run(forwards[1], data_loader_test,  evaluate)
                vali_acc, vali_f1 = self.run(forwards[2], data_loader_vali, evaluate)
            else:
                train_acc, train_f1 = self.run(forwards, data_loader_train, evaluate)
                test_acc, test_f1 = self.run(forwards, data_loader_test,  evaluate)
                vali_acc, vali_f1 = self.run(forwards, data_loader_vali, evaluate)
            # print(loss_other_sum / len(data_loader_train))
            print('Epoch %d/%d : Average Loss %5.4f, Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f'
                  % (e+1, self.cfg.n_epochs, loss_sum / len(data_loader_train), train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1))
            if print_time: print("Training time: %.5f seconds" % (time_sum / len(data_loader_train)))
            if vali_acc > vali_acc_best:
                vali_acc_best = vali_acc
                best_stat = (train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1)
                model_best = copy.deepcopy(model.state_dict())  # copy.deepcopy(model)
                self.save()
        self.model.load_state_dict(model_best)
        print('The Total Epoch have been reached.')
        print('Best Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f' % best_stat)

    def load(self, model_file, load_self=False):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            if self.print_model:
                print('Loading the model from', model_file)
            if load_self:
                self.model.load_self(model_file + '.pt', map_location=self.device)
            else:
                self.model.load_state_dict(torch.load(model_file + '.pt', map_location=self.device), strict=False)

    def save(self, i=0):
        """ save current model """
        if i != 0:
            torch.save(self.model.state_dict(), self.save_path + "_" + str(i) + '.pt')
        else:
            torch.save(self.model.state_dict(),  self.save_path + '.pt')


class TrainerFederated(object):

    def __init__(self, cfg, model, client_num, optimizer, save_path, device, print_model=True):
        self.cfg = cfg
        self.client_num = client_num
        self.model_clients = [copy.deepcopy(model) for i in range(client_num)]
        self.sample_num = [0] * client_num
        self.model_server = copy.deepcopy(model)
        self.optimizers = [optimizer(params=model.parameters(), lr=cfg.lr) for model in self.model_clients]
        self.save_path = save_path
        self.device = device
        self.print_model = print_model
        self.training_stat = []

    def train(self, get_loss, forward, evaluate, data_loader_func, model_file=None, load_self=False):
        self.load(model_file, load_self=load_self)
        num_round = self.cfg.num_round
        print("Federated Training Start.")
        for i in range(num_round):
            print("---Round-%d:---" % (i + 1))
            self.share()
            self.update_clients(get_loss, forward, evaluate, data_loader_func)
            print("---Aggregate-%d:---" % (i + 1))
            self.aggregate_server()
        np.save(os.path.join(self.save_path, 'loss.npy'), np.array(self.training_stat), allow_pickle=False)
        print("Federated Training Finish.")

    def update_clients(self, get_loss, forward, evaluate, data_loader_func, save=True, print_loss=True):
        stat = []
        for c in range(self.client_num):
            model_client = self.model_clients[c].to(self.device)
            optimizer = self.optimizers[c]
            data_loader_train, data_loader_test = data_loader_func(c)
            self.sample_num[c] = len(data_loader_train)
            client_step = 0
            best_loss = 1e6
            best_stat = (0, 0.0, 0.0)
            model_best = model_client.state_dict()
            for e in range(self.cfg.n_epochs):
                loss_sum = 0.0
                time_sum = 0.0
                model_client.train()
                for i, batch in enumerate(data_loader_train):
                    batch = [t.to(self.device) for t in batch]

                    start_time = time.time()
                    optimizer.zero_grad()
                    loss = get_loss(model_client, batch)

                    loss = loss.mean()  # mean() for Data Parallelism
                    loss.backward()
                    optimizer.step()

                    client_step += 1
                    loss_sum += loss.item()
                    time_sum += time.time() - start_time

                loss_train = self.run_client(model_client, forward, data_loader_train, evaluate)
                loss_eva = self.run_client(model_client, forward, data_loader_test, evaluate)
                # print("Train execution time: %.5f seconds" % (time_sum / len(self.data_loader)))
                # print("Train execution time: %.5f seconds" % (time_sum / len(data_loader)))
                if loss_eva < best_loss:
                    best_loss = loss_eva
                    best_stat = (c, loss_train, loss_eva)
                    model_best = copy.deepcopy(model_client.state_dict())  # copy.deepcopy(model)
                    if save:
                        self.save(c)
            model_client.load_state_dict(model_best)
            if print_loss:
                print('Client-%d : Train Loss %5.4f. Test Loss %5.4f' % best_stat)
            stat.append(best_stat[1:])
        print("Clients Update Finish. Average Train Loss %5.4f. Average Test Loss %5.4f"
              % tuple(np.array(stat).mean(axis=0)))
        self.training_stat.append(np.array(stat).mean(axis=0))

    def share(self):
        for model in self.model_clients:
            model.load_state_dict(self.model_server.state_dict())

    def aggregate_server(self):
        dict_server = self.model_server.state_dict()
        for k in dict_server.keys():
            dict_clients = [self.model_clients[i].state_dict()[k].float() for i in range(self.client_num)]
            dict_clients = torch.stack(dict_clients, 0)
            weights_clients = torch.tensor(self.sample_num, dtype=torch.float32) / sum(self.sample_num)
            dict_clients_w = (dict_clients.T * weights_clients.to(self.device)).T
            dict_server[k] = dict_clients_w.sum(0)
            # dict_server[k] = torch.stack(dict_clients, 0).mean(0)
        self.model_server.load_state_dict(dict_server)
        self.save()

    def run_client(self, model, forward, data_loader, evaluate=None):
        """ Evaluation Loop """
        model.eval() # evaluation mode
        model = model.to(self.device)

        results = [] # prediction results
        labels = []
        time_sum = 0.0
        for batch in data_loader:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad(): # evaluation without gradient calculation
                start_time = time.time()
                result, label = forward(model, batch)
                time_sum += time.time() - start_time
                result = result.cpu().numpy()
                results.append(result)
                if label is not None:
                    labels.append(label.cpu().numpy())
        # print("Eval execution time: %.5f seconds" % (time_sum / len(dt)))
        if evaluate:
            return evaluate(np.concatenate(labels, 0), np.concatenate(results, 0))
        else:
            return np.concatenate(results, 0)

    def load(self, model_file, load_self=False):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            if self.print_model:
                print('Loading the model from', model_file)
            if load_self:
                self.model_server.load_self(model_file + '.pt', map_location=self.device)
            else:
                self.model_server.load_state_dict(torch.load(model_file + '.pt', map_location=self.device))

    def save(self, index=None):
        """ save current model """
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        if index is not None:
            torch.save(self.model_clients[index].state_dict(), os.path.join(self.save_path, str(index) + '.pt'))
        else:
            torch.save(self.model_server.state_dict(), os.path.join(self.save_path, 'server.pt'))
            torch.save(self.model_server.state_dict(), self.save_path + '.pt')

