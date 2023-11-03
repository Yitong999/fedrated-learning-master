import numpy as np
import torch
dataset_train_X = np.load('data/colored_mnist/train/images.npy')
dataset_train_Y = np.load('data/colored_mnist/train/attrs.npy')
#torch.concat([dataset_train_X,dataset_train_Y],dim=0)
print(dataset_train_Y.shape)
#print(dataset_train.shape)
train_target_attr = []
for data in dataset_train_Y:
    train_target_attr.append(int(data[-2]))
train_target_attr = torch.LongTensor(train_target_attr)

attr_dims = []
attr_dims.append(torch.max(train_target_attr).item() + 1)

print(attr_dims[0])