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


import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--num_attackers', type=int, default=0, help="number of attackers: f")
    parser.add_argument('--frac', type=float, default=1.0, help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--local_iter', type=int, default=30, help="local iteration(number of batch)")
    parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')

    # sampling arguments
    parser.add_argument('--single', action='store_true', help="assign single digits to each user")
    parser.add_argument('--fix_total', action='store_true', help="fix total users to 100")

    # model arguments
    parser.add_argument('--model', type=str, default='URLNet', required=True, help='model name')

    # attack arguments
    parser.add_argument('--use_normal', type=int, default=0, help='perform gaussian attack on n users')
    parser.add_argument('--normal_scale', type=float, default=100.0, help='scale of noise in percent')
    parser.add_argument('--attacker_ep', type=int, default=5, help="the number of attacker's local epochs: E")
    parser.add_argument('--change_rate', type=float, default=-1.0, help='scale of noise in percent')
    parser.add_argument('--use_poison', type=int, default=-1, help='perform poison attack on n users')
    parser.add_argument('--attack_label', type=int, default=-1, help='select the label to be attacked in poisoning attack')
    parser.add_argument('--donth_attack', action='store_true', help='this attack excludes the selected nodes from aggregation')

    # backdoor attack arguments
    parser.add_argument('--is_backdoor',type=bool, default=False, help="use backdoor attack")
    parser.add_argument('--backdoor_per_batch', type=int,default=20, help="poisoned data during training per batch")
    parser.add_argument('--backdoor_scale_factor', type=float, default=1.0, help="scale factor for local model's parameters")
    parser.add_argument('--backdoor_label', type=int, default=-1, help="target label for backdoor attack")
    parser.add_argument('--backdoor_single_shot_scale_epoch', type=int, default=-1, help="used for one-single-shot; -1 means no single scaled shot")

    # aggregation arguments
    parser.add_argument('--agg', type=str, default='average', choices=['average', 'median', 'trimmed_mean', 'repeated',
                                                                       'irls', 'simple_irls','irls_median', 'irls_theilsen',
                                                                       'irls_gaussian', 'fg'], help="Aggregation methods")
    parser.add_argument('--Lambda', type=float, default=2.0, help='set lambda of irls (default: 2.0)')
    parser.add_argument('--thresh', type=float, default=0.5, help='set thresh of irls restriction (default: 0.5)')
    parser.add_argument('--alpha', type=float, default=0.2, help='set thresh of trimmed mean (default: 0.2)')
    parser.add_argument('--use_memory', type=str2bool, default=True, help="use FoolsGold memory option")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', type=int, default=0, help='whether i.i.d or not, 1 for iid, 0 for non-iid')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID")
    parser.add_argument('--verbose', type=int, default=0, help='verbose print, 1 for True, 0 for False')
    parser.add_argument('--seed', type=int, default=1237, help='random seed (default: 1234)')

    # our arguments
    parser.add_argument('--kappa', type=float, default=0.3, required=False, help='weight for positive observation of the objective function applied to model when we consider R and S')
    parser.add_argument('--eta', type=float, default=0.7, required=False, help='eta + k = 1')
    parser.add_argument('--W', type=float, default=2, required=False, help='non-information prior weight is the weight of the reputation we assigned initially, w=2 assigned in paper by default')
    parser.add_argument('--a', type=float, default=0.5, required=False, help='a priori probability in the absence of committed belief mass. If we increase a, we scale up reputation')
    parser.add_argument('--z', type=float, default=0.5, required=False, help='time decay or interaction freshness z (0,1). If we increase z we scale up our reputation model')
    arser.add_argument('--s', type=float, default=10, required=False, help='window length')

    parser.add_argument('--reputation_active', type=bool, default=False, required=False,
                        help='whether to use our reputation model')
    parser.add_argument('--reputation_effect', default=[], required=False, help='reputation array to store history of reputation scores')
    parser.add_argument('--reputation_active_type', type=int, default=0, choices=[0, 1], required=False,
                        help='choose type of reputation model'
                             '0: stands for Subjective logic')
    parser.add_argument('--cloud', type=bool, default=True, required=False,
                        help='choose location of dataset')

    args = parser.parse_args()
    return args
