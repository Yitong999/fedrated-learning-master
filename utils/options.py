#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=1000, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=2, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_bs', type=int, default=12, help="local train batch size: B")
    parser.add_argument('--bs', type=int, default=256, help="test batch size")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument("--num_steps", help="local iterations size: B", default=5, type=int)

    # other arguments
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')

    ##
    # training
    parser.add_argument("--lr", help='learning rate', default=1e-2, type=float)
    parser.add_argument("--weight_decay", help='weight_decay', default=0.0, type=float)
    parser.add_argument("--momentum", help='momentum', default=0.9, type=float)
    parser.add_argument("--num_workers", help="workers number", default=0, type=int)
    parser.add_argument("--exp", help='experiment name', default='debugging', type=str)
    parser.add_argument("--device", help="cuda or cpu", default='mps', type=str)
    parser.add_argument("--target_attr_idx", help="target_attr_idx", default=0, type=int)
    parser.add_argument("--bias_attr_idx", help="bias_attr_idx", default=1, type=int)
    parser.add_argument("--dataset", help="data to train, ", default='images', type=str)
    parser.add_argument("--percent", help="percentage of conflict", default="1pct", type=str)
    parser.add_argument("--use_lr_decay", action='store_true', help="whether to use learning rate decay")
    parser.add_argument("--lr_decay_step", help="learning rate decay steps", type=int, default=10000)
    parser.add_argument("--q", help="GCE parameter q", type=float, default=0.7)
    parser.add_argument("--lr_gamma", help="lr gamma", type=float, default=0.1)
    parser.add_argument("--lambda_dis_align", help="lambda_dis in Eq.2", type=float, default=1.0)
    parser.add_argument("--lambda_swap_align", help="lambda_swap_b in Eq.3", type=float, default=1.0)
    parser.add_argument("--lambda_swap", help="lambda swap (lambda_swap in Eq.4)", type=float, default=1.0)
    parser.add_argument("--ema_alpha", help="use weight mul", type=float, default=0.7)
    parser.add_argument("--curr_step", help="curriculum steps", type=int, default=0)
    parser.add_argument("--use_type0", action='store_true', help="whether to use type 0 CIFAR10C")
    parser.add_argument("--use_type1", action='store_true', help="whether to use type 1 CIFAR10C")
    parser.add_argument("--use_resnet20", help="Use Resnet20",
                        action="store_true")  # ResNet 20 was used in Learning From Failure CifarC10 (We used ResNet18 in our paper)
    parser.add_argument("--model", help="which network, [MLP, ResNet18, ResNet20, ResNet50]", default='MLP', type=str)

    # logging
    parser.add_argument("--log_dir", help='path for saving model', default='./log', type=str)
    parser.add_argument("--data_dir", help='path for loading data', default='dataset', type=str)
    parser.add_argument("--valid_freq", help='frequency to evaluate on valid/test set', default=500, type=int)
    parser.add_argument("--log_freq", help='frequency to log on tensorboard', default=500, type=int)
    parser.add_argument("--save_freq", help='frequency to save model checkpoint', default=1000, type=int)
    parser.add_argument("--wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--tensorboard", action="store_true", help="whether to use tensorboard")

    # experiment
    parser.add_argument("--train_ours", action="store_true", help="whether to train our method")
    parser.add_argument("--train_vanilla", action="store_true", help="whether to train vanilla")
    args = parser.parse_args()
    return args
