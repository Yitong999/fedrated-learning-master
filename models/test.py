#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.Update import DatasetSplit
from torch.utils.data import Dataset, DataLoader
from functools import partial
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def concat_dummy(z):
    def hook(model, input, output):
        z.append(output.squeeze())
        return torch.cat((output, torch.zeros_like(output)), dim=1)
    return hook

def evaluate_ours( args, model_b, model_l, x_test,y_test, model='label'):

    test_loader = DataLoader(
        CustomDataset(x_test,y_test),
        batch_size= args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model_b.eval()
    model_l.eval()

    total_correct, total_num = 0, 0

    for idx, (data, attr) in enumerate(test_loader):
        label = attr[:, 0]
        # label = attr
        data = data.to(args.device)
        label = label.to(args.device)

        with torch.no_grad():
            z_l, z_b = [], []
            data = torch.transpose(data, 1, 3)
            hook_fn = model_l.avgpool.register_forward_hook(concat_dummy(z_l))
            _ = model_l(data)
            hook_fn.remove()
            z_l = z_l[0]
            hook_fn = model_b.avgpool.register_forward_hook(concat_dummy(z_b))
            _ = model_b(data)
            hook_fn.remove()
            z_b = z_b[0]
            z_origin = torch.cat((z_l, z_b), dim=1)
            if model == 'bias':
                pred_label = model_b.fc(z_origin)
            else:
                pred_label = model_l.fc(z_origin)
            pred = pred_label.data.max(1, keepdim=True)[1].squeeze(1)
            correct = (pred == label).long()
            total_correct += correct.sum()
            total_num += correct.shape[0]

    accs = total_correct / float(total_num)

    return accs

