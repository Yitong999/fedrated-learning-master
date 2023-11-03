#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import wandb
import torch
from torch import nn, autograd, optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from tqdm import tqdm

from data.util import IdxDataset
from models.loss import GeneralizedCELoss
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class DatasetSplit(Dataset):
    def __init__(self, dataset,target, idxs):
        self.dataset = dataset
        self.target = target
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]],self.target[self.idxs[item]]
        return image, label

def DatasetSplittarget(Dataset):
    train_target_attr = []
    for idx,(data,target) in enumerate(Dataset):
        train_target_attr.append(target[0])
    train_target_attr = torch.LongTensor(train_target_attr)
    attr_dims = []
    attr_dims.append(torch.max(train_target_attr).item() + 1)
    return attr_dims[0]

class EMA:
    def __init__(self, num_classes, alpha=0.9):
        self.alpha = alpha
        self.num_classes = num_classes
        self.loss_ema = {i: 0 for i in range(num_classes)}  # 初始化每个类别的EMA损失为0
        self.max_loss_dict = {i: 0 for i in range(num_classes)}  # 初始化每个类别的最大损失为0

    def update(self, loss, label):
        # 更新EMA
        self.loss_ema[label] = self.alpha * self.loss_ema[label] + (1 - self.alpha) * loss
        # 更新最大损失
        self.max_loss_dict[label] = max(self.max_loss_dict[label], self.loss_ema[label])

    def get_max_loss(self, label):
        return self.max_loss_dict.get(label, 0)  # 如果没有记录，返回0

class LocalUpdate(object):
    def __init__(self, args, x_dataset=None,y_target = None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.train_loader = DataLoader(
            DatasetSplit(x_dataset,y_target, idxs),
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        # define loss
        # logging directories
        self.log_dir = os.path.join(args.log_dir, args.dataset, args.exp)
        self.summary_dir = os.path.join(args.log_dir, args.dataset, "summary", args.exp)
        self.summary_gradient_dir = os.path.join(self.log_dir, "gradient")
        self.result_dir = os.path.join(self.log_dir, "result")
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.args.device)
        self.bias_criterion = nn.CrossEntropyLoss(reduction='none').to(self.args.device)

        print(f'self.criterion: {self.criterion}')
        print(f'self.bias_criterion: {self.bias_criterion}')
        self.train_dataset = IdxDataset(DatasetSplit(x_dataset,y_target, idxs))
        self.num_classes = DatasetSplittarget(DatasetSplit(x_dataset,y_target, idxs))
        # self.sample_loss_ema_b = EMA(torch.LongTensor(self.train_target_attr), num_classes=self.num_classes,
        #                              alpha=args.ema_alpha)
        # self.sample_loss_ema_d = EMA(torch.LongTensor(self.train_target_attr), num_classes=self.num_classes,
        #                              alpha=args.ema_alpha)

        #print(f'alpha : {self.sample_loss_ema_d.alpha}')

        self.best_valid_acc_b, self.best_test_acc_b = 0., 0.
        self.best_valid_acc_d, self.best_test_acc_d = 0., 0.
        print('finished model initialization....')

    def save_ours(self, step, best=None):
        if best:
            model_path = os.path.join(self.result_dir, "best_model_l.th")
        else:
            model_path = os.path.join(self.result_dir, "model_l_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model_l.state_dict(),
            'optimizer': self.optimizer_l.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        if best:
            model_path = os.path.join(self.result_dir, "best_model_b.th")
        else:
            model_path = os.path.join(self.result_dir, "model_b_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model_b.state_dict(),
            'optimizer': self.optimizer_b.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        print(f'{step} model saved ...')

    def board_ours_loss(self, step, loss_dis_conflict, loss_dis_align, loss_swap_conflict, loss_swap_align, lambda_swap):

        if self.args.wandb:
            wandb.log({
                "loss_dis_conflict":    loss_dis_conflict,
                "loss_dis_align":       loss_dis_align,
                "loss_swap_conflict":   loss_swap_conflict,
                "loss_swap_align":      loss_swap_align,
                "loss":                 (loss_dis_conflict + loss_dis_align) + lambda_swap * (loss_swap_conflict + loss_swap_align)
            }, step=step,)

        if self.args.tensorboard:
            self.writer.add_scalar(f"loss/loss_dis_conflict",  loss_dis_conflict, step)
            self.writer.add_scalar(f"loss/loss_dis_align",     loss_dis_align, step)
            self.writer.add_scalar(f"loss/loss_swap_conflict", loss_swap_conflict, step)
            self.writer.add_scalar(f"loss/loss_swap_align",    loss_swap_align, step)
            self.writer.add_scalar(f"loss/loss",               (loss_dis_conflict + loss_dis_align) + lambda_swap * (loss_swap_conflict + loss_swap_align), step)

    def concat_dummy(self, z):
        def hook(model, input, output):
            z.append(output.squeeze())
            return torch.cat((output, torch.zeros_like(output)), dim=1)
        return hook


    def trians(self,model_l,model_b):
       model_l.to(self.args.device)
       model_b.to(self.args.device)

       optimizer_l = torch.optim.Adam(
            model_l.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
       optimizer_b = torch.optim.Adam(
           model_b.parameters(),
           lr=self.args.lr,
           weight_decay=self.args.weight_decay,
       )
       if self.args.use_lr_decay:
            scheduler_b = optim.lr_scheduler.StepLR(optimizer_b, step_size=self.args.lr_decay_step,
                                                         gamma=self.args.lr_gamma)
            scheduler_l = optim.lr_scheduler.StepLR(optimizer_l, step_size=self.args.lr_decay_step,
                                                          gamma=self.args.lr_gamma)
       self.bias_criterion = GeneralizedCELoss(q=0.7)

       for step in tqdm(range(self.args.num_steps)):
            sample_loss_ema_d = EMA(num_classes=self.num_classes, alpha=self.args.ema_alpha)
            sample_loss_ema_b = EMA(num_classes=self.num_classes, alpha=self.args.ema_alpha)
            for index,(data,target) in enumerate(self.train_loader):
                data = data.float().to(self.args.device)
                attr = target.to(self.args.device)
                label = attr[:,1].to(self.args.device)

                # Feature extraction
                # Prediction by concatenating zero vectors (dummy vectors).
                # We do not use the prediction here.

                z_b = []
                # Use this only for reproducing CIFARC10 of LfF
                if self.args.use_resnet20:
                    hook_fn = model_b.layer3.register_forward_hook(self.concat_dummy(z_b))
                    data = torch.transpose(data, 1, 3)
                    _ = model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]

                    z_l = []
                    hook_fn = model_l.layer3.register_forward_hook(self.concat_dummy(z_l))
                    _ = model_l(data)
                    hook_fn.remove()

                    z_l = z_l[0]

                else:
                    hook_fn = model_b.avgpool.register_forward_hook(self.concat_dummy(z_b))
                    data = torch.transpose(data, 1, 3)
                    _ = model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]

                    z_l = []
                    hook_fn = model_l.avgpool.register_forward_hook(self.concat_dummy(z_l))
                    _ = model_l(data)
                    hook_fn.remove()

                    z_l = z_l[0]

                # Gradients of z_b are not backpropagated to z_l (and vice versa) in order to guarantee disentanglement of representation.
                z_conflict = torch.cat((z_l, z_b.detach()), dim=1)
                z_align = torch.cat((z_l.detach(), z_b), dim=1)

                # Prediction using z=[z_l, z_b]
                pred_conflict = model_l.fc(z_conflict)
                pred_align = model_b.fc(z_align)

                loss_dis_conflict = self.criterion(pred_conflict, label).detach()
                loss_dis_align = self.criterion(pred_align, label).detach()

                # 第1步: 更新每个类别的EMA和最大损失
                for c in range(self.num_classes):
                    class_index = torch.where(label == c)[0]
                    if len(class_index) > 0:  # 检查是否有该类别的样本
                        sample_loss_ema_d.update(loss_dis_conflict[class_index].mean().item(), c)
                        sample_loss_ema_b.update(loss_dis_align[class_index].mean().item(), c)

                # 第2步: 对每个类别的损失进行归一化
                normalized_loss_dis_conflict = torch.zeros_like(loss_dis_conflict)
                normalized_loss_dis_align = torch.zeros_like(loss_dis_align)

                for c in range(self.num_classes):
                    class_index = torch.where(label == c)[0]
                    if len(class_index) > 0:
                        max_loss_conflict = sample_loss_ema_d.get_max_loss(c)
                        max_loss_align = sample_loss_ema_b.get_max_loss(c)
                        normalized_loss_dis_conflict[class_index] = loss_dis_conflict[class_index] / (
                                    max_loss_conflict + 1e-8)
                        normalized_loss_dis_align[class_index] = loss_dis_align[class_index] / (max_loss_align + 1e-8)

                # 第3步: 根据两种不同损失的比例计算损失的权重
                loss_weight = normalized_loss_dis_align / (
                            normalized_loss_dis_align + normalized_loss_dis_conflict + 1e-8)

                # 至此，`loss_weight`现在包含了根据你的加权公式计算出的每个样本的损失权重。

                loss_dis_conflict = self.criterion(pred_conflict, label) * loss_weight.to(
                    self.args.device)  # Eq.2 W(z)CE(C_i(z),y)
                loss_dis_align = self.bias_criterion(pred_align, label)  # Eq.2 GCE(C_b(z),y)

                # feature-level augmentation : augmentation after certain iteration (after representation is disentangled at a certain level)
                if step > self.args.curr_step:
                    indices = np.random.permutation(z_b.size(0))
                    z_b_swap = z_b[indices]         # z tilde
                    label_swap = label[indices]     # y tilde

                    # Prediction using z_swap=[z_l, z_b tilde]
                    # Again, gradients of z_b tilde are not backpropagated to z_l (and vice versa) in order to guarantee disentanglement of representation.
                    z_mix_conflict = torch.cat((z_l, z_b_swap.detach()), dim=1)
                    z_mix_align = torch.cat((z_l.detach(), z_b_swap), dim=1)

                    # Prediction using z_swap
                    pred_mix_conflict = model_l.fc(z_mix_conflict)
                    pred_mix_align = model_b.fc(z_mix_align)

                    loss_swap_conflict = self.criterion(pred_mix_conflict, label) * loss_weight.to(self.args.device)     # Eq.3 W(z)CE(C_i(z_swap),y)
                    loss_swap_align = self.bias_criterion(pred_mix_align, label_swap)                               # Eq.3 GCE(C_b(z_swap),y tilde)
                    lambda_swap = self.args.lambda_swap                                                             # Eq.3 lambda_swap_b

                else:
                    # before feature-level augmentation
                    loss_swap_conflict = torch.tensor([0]).float()
                    loss_swap_align = torch.tensor([0]).float()
                    lambda_swap = 0

            


            loss_dis  = loss_dis_conflict.mean() + self.args.lambda_dis_align * loss_dis_align.mean()                # Eq.2 L_dis
            loss_swap = loss_swap_conflict.mean() + self.args.lambda_swap_align * loss_swap_align.mean()             # Eq.3 L_swap
            loss = loss_dis + lambda_swap * loss_swap                                                           # Eq.4 Total objective
            optimizer_l.zero_grad()
            optimizer_b.zero_grad()
            loss.backward()
            optimizer_l.step()
            optimizer_b.step()

            if step >= self.args.curr_step and self.args.use_lr_decay:
                scheduler_b.step()
                scheduler_l.step()

            if self.args.use_lr_decay and step % self.args.lr_decay_step == 0:
                print('******* learning rate decay .... ********')
                print(f"self.optimizer_b lr: {optimizer_b.param_groups[-1]['lr']}")
                print(f"self.optimizer_l lr: {optimizer_l.param_groups[-1]['lr']}")

            # avg_acc = evaluate_ours( args, model_b, model_l, dataset_test_X, dataset_test_Y , model='label')
            # print("epoch",iter,"acc:",avg_acc)


       return model_l.state_dict(),model_b.state_dict()
