#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

from models.util import get_model

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Fed import FedAvg
from models.test import  evaluate_ours
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.dataset == 'images':
        dataset_train_X = np.load('data/colored_mnist/train/images.npy')
        dataset_test_X = np.load('data/colored_mnist/valid/images.npy')
        dataset_train_Y = np.load('data/colored_mnist/train/attrs.npy')
        dataset_test_Y = np.load('data/colored_mnist/valid/attrs.npy')
        dict_users = mnist_iid(dataset_train_X, args.num_users)

    else:
        exit('Error: unrecognized dataset')

    if args.use_resnet20:  # Use this option only for comparing with LfF
        model_l_glob = get_model('ResNet20_OURS',args.num_classes).to(args.device)
        model_b_glob = get_model('ResNet20_OURS', args.num_classes).to(args.device)
        print('our resnet20....')
    else:
       model_l_glob = get_model('mlp_DISENTANGLE',args.num_classes).to(args.device)
       model_b_glob = get_model('mlp_DISENTANGLE', args.num_classes).to(args.device)

    model_l_glob.to(args.device)
    model_b_glob.to(args.device)
    model_l_glob.train()
    model_b_glob.train()

    # copy weights
    w_l_glob_weight = model_l_glob.state_dict()
    w_b_glob_weight = model_b_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    for iter in range(args.epochs):
        loss_l_locals = []
        loss_b_locals = []
        model_l_glob.train()
        model_b_glob.train()
        w_l_locals=[]
        w_b_locals=[]
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local = LocalUpdate(args=args, x_dataset=dataset_train_X,y_target =dataset_train_Y , idxs=dict_users[idx])
            w_l, w_b = local.trians(copy.deepcopy(model_l_glob),copy.deepcopy(model_b_glob))

            #w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_l_locals.append(copy.deepcopy(w_l))
            w_b_locals.append(copy.deepcopy(w_b))

        # update global weights
        w_glob_l,w_glob_b = FedAvg(w_l_locals,w_b_locals)

        # copy weight to net_glob
        model_l_glob.load_state_dict(w_glob_l)
        model_b_glob.load_state_dict(w_glob_b)
        avg_acc = evaluate_ours( args, model_b_glob, model_l_glob,dataset_test_X,dataset_test_Y , model='label')
        print("epoch",iter,"acc:",avg_acc)
