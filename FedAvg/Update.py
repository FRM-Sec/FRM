#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Copyright (C) 2021 - 2022
# @author IMDEA NETWORKS
#
# This file is part of the FRM framework
#
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses>.

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn import metrics
import pickle


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, attack_label=-1):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.attack_label = attack_label
        if self.attack_label >= 0:
            if self.attack_label != 0 and self.attack_label != 1 and self.attack_label != 3:
                print(
                    'currently attack label only supports 0 for URL dataset, 1 for mnist and 3 (Cat) for cifar, not {}'.format(
                    self.attack_label))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[int(self.idxs[item])]
        if self.attack_label >= 0:
            if label == self.attack_label == 1 and self.dataset == 'mnist':
                label = (label + 6) % 10
            elif label == self.attack_label == 1 and self.dataset == 'URL':
                label = 5
            elif label == self.attack_label == 3:   # cat to dog for cifar
                label = label + 2
            elif label == self.attack_label == 0:  # URL dataset
                label = label + 5

        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, tb, attack_label=-1, backdoor_label=-1, test_flag=False):
        self.args = args
        self.attack_label = attack_label
        self.backdoor_label = backdoor_label
        if self.attack_label >= 0 and self.backdoor_label >= 0: # if the args has something wrong
            self.attack_label = -1  # make sure that only perform one kind of attack
        self.backdoor_pixels = [[0, 0], [0, 1], [0, 2],
                                [0, 4], [0, 5], [0, 6],
                                [2, 0], [2, 1], [2, 2],
                                [2, 4], [2, 5], [2, 6],
                                ]  # temporarily hard code
        # backdoor attack for URL
        try:
            with open ('back_door.pkl', 'rb') as f:
                self.backdoor_URL = pickle.load(f)
        except Exception as ex:
            print(ex)

        if args.model == 'mobilenet' or args.model == 'loannet':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_func = nn.NLLLoss()
        self.tb = tb
        self.dis_loss = nn.L1Loss()

        if test_flag: # test_flag -> idxs all for test; otherwise only 20% of idxs for test
            if backdoor_label >= 0: # backdoor attack will poison all data in training..
                self.ldr_test = DataLoader(DatasetSplit(dataset, idxs, -1), batch_size=args.local_bs,
                                           shuffle=True)
            else: # label-flipping attack
                self.ldr_test = DataLoader(DatasetSplit(dataset, idxs, self.attack_label), batch_size=args.local_bs,
                                                   shuffle=True)
            self.ldr_train=None
            self.ldr_val=None
        else:
            self.ldr_train, self.ldr_val, self.ldr_test = self.train_val_test(dataset, list(idxs))

    def train_val_test(self, dataset, idxs):
        # split train, validation, and test
        train_split = round(len(idxs) * 0.7)
        test_split = round(len(idxs) * 0.8)
        idxs_train = idxs[:int(train_split)]
        idxs_val = idxs[int(train_split):int(test_split)]
        idxs_test = idxs[int(test_split):]
        if self.backdoor_label >= 0:  # backdoor attack will poison part of data per batch in training..
            train = DataLoader(DatasetSplit(dataset, idxs_train, -1), batch_size=self.args.local_bs,
                               shuffle=True)
            val = DataLoader(DatasetSplit(dataset, idxs_val, -1), batch_size=int(len(idxs_val) / 10),
                             shuffle=True)
            test = DataLoader(DatasetSplit(dataset, idxs_test, -1), batch_size=int(len(idxs_test) / 10),
                              shuffle=True)
        else:
            train = DataLoader(DatasetSplit(dataset, idxs_train, self.attack_label), batch_size=self.args.local_bs,
                               shuffle=True)
            val = DataLoader(DatasetSplit(dataset, idxs_val, self.attack_label), batch_size=int(len(idxs_val) / 10),
                             shuffle=True)
            test = DataLoader(DatasetSplit(dataset, idxs_test, self.attack_label), batch_size=int(len(idxs_test) / 10),
                              shuffle=True)
        return train, val, test

    def update_weights(self, net):
        global_net = dict()
        if self.backdoor_label >= 0:  # backdoor attack can scale, so it needs the original value of parameters
            for name, data in net.state_dict().items():
                global_net[name] = net.state_dict()[name].clone()
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # if any(labels == 1):
                #     print(labels)
                #     exit(-1)
                if batch_idx > self.args.local_iter > 0:
                    break
                if self.backdoor_label >= 0:  # backdoor attack
                    images, labels, poison_count = self.get_poison_batch(images, labels,
                                                                         self.args.backdoor_per_batch,
                                                                         self.backdoor_label,
                                                                         evaluation=False)
                if self.args.gpu != -1:
                    images, labels = images.cuda(), labels.cuda()
                if self.args.dataset== 'URL':#@TODO need to clear out that later
                    images=images.float()
                    labels=labels.long()

                images, labels = autograd.Variable(images), autograd.Variable(labels)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.gpu != -1:
                    loss = loss.cpu()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                self.tb.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        if self.backdoor_label >= 0:  # backdoor attack can scale
            for name, data in net.state_dict().items():
                new_value = global_net[name] + (data - global_net[name]) * self.args.backdoor_scale_factor
                net.state_dict()[name].copy_(new_value)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), net

    def update_gradients(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        client_grad = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if batch_idx > self.args.local_iter > 0:
                    break
                if self.backdoor_label >= 0:  # backdoor attack
                    images, labels, poison_count = self.get_poison_batch(images, labels,
                                                                         self.args.backdoor_per_batch,
                                                                         self.backdoor_label,
                                                                         evaluation=False)
                if self.args.gpu != -1:
                    images, labels = images.cuda(), labels.cuda()
                if self.args.dataset == 'URL':#@TODO need to clear out that later
                    images = images.float()
                    labels = labels.long()
                images, labels = autograd.Variable(images), autograd.Variable(labels)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                # get gradients
                for i, (name, params) in enumerate(net.named_parameters()):
                    if params.requires_grad:
                        if iter == 0 and batch_idx == 0:
                            client_grad.append(params.grad.clone())
                        else:
                            client_grad[i] += params.grad.clone()

                optimizer.step()
                if self.args.gpu != -1:
                    loss = loss.cpu()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                self.tb.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        # there is no scale for backdoor attack when update gradients, because
        # foolsgold paper said it doesn't consider scaling malicious updates to overpower honest clients

        if self.backdoor_label >= 0:  # backdoor attack can scale
            for i, (name, params) in enumerate(net.named_parameters()):
                if params.requires_grad:
                    client_grad[i]*= self.args.backdoor_scale_factor
        return client_grad, sum(epoch_loss) / len(epoch_loss),net

    def add_dis_to_gradient(self, net, w_glob, alpha=0.9):
        total_dis = 0
        keys = list(w_glob.keys())
        for i, p in enumerate(net.parameters()):
            key = keys[i]
            dis = p - w_glob[key]
            p.grad = p.grad * alpha + dis * (1 - alpha) * 100
            total_dis += dis.abs().sum()
        return total_dis

    def update_weights_with_constrain(self, net, w_glob):
        global_net = dict()
        if self.backdoor_label >= 0:
            for name, data in net.state_dict().items():
                global_net[name] = net.state_dict()[name].clone()

        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if batch_idx > self.args.local_iter > 0:
                    break
                if self.backdoor_label >= 0:  # backdoor attack
                    images, labels, poison_count = self.get_poison_batch(images, labels,
                                                                         self.args.backdoor_per_batch,
                                                                         self.backdoor_label,
                                                                         evaluation=False)
                if self.args.gpu != -1:
                    images, labels = images.cuda(), labels.cuda()
                if self.args.dataset == 'URL':#@TODO need to clear out that later
                    images=images.float()
                    labels=labels.long()
                images, labels = autograd.Variable(images), autograd.Variable(labels)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                # after_weight = net.state_dict()
                # dis = []
                # for k in w_glob.keys():
                #     dis.append((after_weight[k] - w_glob[k]).flatten())
                # dis = torch.cat(dis, dim=0)
                # param_size = len(dis)
                # d_loss = dis.abs().sum() / param_size
                before_loss = loss.item()
                # loss = loss * alpha + (1.0 - alpha ) * d_loss
                # print('Added model distance loss: {:6f}, train loss {:6f}'.format(d_loss, before_loss))

                loss.backward(retain_graph=True)
                total_dis = self.add_dis_to_gradient(net, w_glob)
                optimizer.step()
                if self.args.gpu != -1:
                    loss = loss.cpu()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDistance: {:6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), before_loss, total_dis))
                self.tb.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        if self.backdoor_label >= 0:
            for name, data in net.state_dict().items():
                new_value = global_net[name] + (data - global_net[name]) * self.args.backdoor_scale_factor
                net.state_dict()[name].copy_(new_value)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def test(self, net):
        acc = 0.0
        total_len = 0
        for batch_idx, (images, labels) in enumerate(self.ldr_test):
            if self.args.gpu != -1:
                images, labels = images.cuda(), labels.cuda()
            if self.args.dataset == 'URL':#@TODO need to clear out that later
                images = images.float()
                labels = labels.long()
            images, labels = autograd.Variable(images), autograd.Variable(labels)
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            if self.args.gpu != -1:
                loss = loss.cpu()
                log_probs = log_probs.cpu()
                labels = labels.cpu()
            y_pred = np.argmax(log_probs.data, axis=1)
            acc += metrics.accuracy_score(y_true=labels.data, y_pred=y_pred) * len(labels)
            total_len += len(labels)
        acc /= float(total_len)
        return acc, loss.item()

    def backdoor_test(self, net):
        if self.backdoor_label < 0:
            return None, None
        acc = 0.0
        total_len = 0
        for batch_idx, (images, labels) in enumerate(self.ldr_test):
            images, labels, poison_count = self.get_poison_batch(images, labels,
                                                                 self.args.backdoor_per_batch, self.backdoor_label,
                                                                 evaluation=True)
            if self.args.dataset == 'URL':#@TODO need to clear out that later
                images = images.float()
                labels = labels.long()
            if self.args.gpu != -1:
                images, labels = images.cuda(), labels.cuda()
            images, labels = autograd.Variable(images), autograd.Variable(labels)
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            if self.args.gpu != -1:
                loss = loss.cpu()
                log_probs = log_probs.cpu()
                labels = labels.cpu()
            y_pred = np.argmax(log_probs.data, axis=1)
            acc += metrics.accuracy_score(y_true=labels.data, y_pred=y_pred) * len(labels)
            total_len += len(labels)
        acc /= float(total_len)
        return acc, loss.item()

    def get_probs(self, net):
        all_probs = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                if self.args.gpu != -1:
                    images, labels = images.cuda(), labels.cuda()
                log_probs = net(images)
                probs = torch.exp(log_probs)
                all_probs += probs.gather(1, labels.view(-1, 1)).sum()
        return all_probs

    def get_poison_batch(self, images, targets, backdoor_per_batch, backdoor_label, evaluation=False):
        poison_count = 0
        new_images = torch.empty(images.shape)
        new_targets = torch.empty(targets.shape)

        for index in range(0, len(images)):
            if evaluation:  # will poison all batch data when testing
                new_targets[index] = backdoor_label
                new_images[index] = self.add_backdoor_pixels(images[index])
                poison_count += 1

            else:  # will poison backdoor_per_batch data when training
                if index < backdoor_per_batch:
                    new_targets[index] = backdoor_label
                    new_images[index] = self.add_backdoor_pixels(images[index])
                    poison_count += 1
                else:
                    new_images[index] = images[index]
                    new_targets[index] = targets[index]

        new_targets = new_targets.long()
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images, new_targets, poison_count

    def add_backdoor_pixels(self, image):
        if self.args.dataset == 'URL':# @TODO need to clear out that later
            for i in range(0, len(self.backdoor_URL)):
                name=self.backdoor_URL[i][0]
                value=self.backdoor_URL[i][1]
                index = self.backdoor_URL[i][2]
                image[index]=value
        else:
            if image.shape[0] == 3:
                for i in range(0, len(self.backdoor_pixels)):
                    pos = self.backdoor_pixels[i]
                    image[0][pos[0]][pos[1]] = 1
                    image[1][pos[0]][pos[1]] = 1
                    image[2][pos[0]][pos[1]] = 1
            elif image.shape[0] == 1:
                for i in range(0, len(self.backdoor_pixels)):
                    pos = self.backdoor_pixels[i]
                    image[0][pos[0]][pos[1]] = 1
        return image
