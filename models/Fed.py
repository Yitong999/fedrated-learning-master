#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w_l,w_b):
    w_avg_l = copy.deepcopy(w_l[0])
    for k in w_avg_l.keys():
        for i in range(1, len(w_l)):
            w_avg_l[k] += w_l[i][k]
        w_avg_l[k] = torch.div(w_avg_l[k], len(w_l))

    w_avg_b = copy.deepcopy(w_b[0])
    for k in w_avg_b.keys():
        for i in range(1, len(w_b)):
            w_avg_b[k] += w_b[i][k]
        w_avg_b[k] = torch.div(w_avg_b[k], len(w_b))

    return w_avg_l,w_avg_l
