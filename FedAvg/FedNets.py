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
from torch import nn
import torch.nn.functional as F
from resnet import ResNet18

DEVICE = torch.device("cuda:0" if torch.cuda.is_available () else "cpu")


def build_model(args):
    if args.model == 'smallcnn' and args.dataset == 'mnist':
        net_glob = SmallCNNMnist(args=args)
    elif args.dataset == 'cifar':
        net_glob = ResNet18(args)
    elif args.model == 'URLNet' and args.dataset == 'URL':
        net_glob = URLNet().to(DEVICE)
    else:
        print(args.model)
        print(args.dataset)
        exit('Error: unrecognized model for these parameters')

    if args.gpu != -1:
        net_glob = net_glob.cuda()
    return net_glob


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class SmallCNNMnist(nn.Module):
    def __init__(self, args):
        super(SmallCNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 4, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4 * 4 * 8, 16)
        self.fc2 = nn.Linear(16, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class URLNet(nn.Module):
    """
    Simple NN for URL dataset
    """
    def __init__(self, in_dim=1000, n_hidden_1=256, n_hidden_2=32, out_dim=6):
        super(URLNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),
                                    nn.Dropout(0.5), # drop 50% of the neuron to avoid over-fitting
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.Dropout(0.5),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # if np.isnan(np.sum(x.data.cpu().numpy())):
        #     raise ValueError()
        return F.log_softmax (x, dim=1)
