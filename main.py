import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from model import CNN1
from dataset import MyDataset
from trainer import train, test
from pathlib import Path
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt


if __name__ == '__main__':

    # set path
    PATH=Path('/home/aims/Dropbox/AMMI/Tutorial/NN_1/project/Cat_Dog_data')
    TRAIN =Path(PATH/'train')
    VALID = Path(PATH/'test')

    # set hyperparameter
    num_workers = 0
    batch_size = 32
    n_features = 6
    input_size = 224
    output_size = 2

    # create custom tranform
    train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    # Loading the data using custom data loader
    train_data = MyDataset(TRAIN, transform=train_transforms)
    valid_data = MyDataset(VALID,transform=valid_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,  num_workers=num_workers,shuffle=True)


    # Instantiate mode
    model_cnn = CNN1(input_size, n_features, output_size)
    optimizer = optim.SGD(model_cnn.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(5):
        train(epoch, model_cnn, train_loader, optimizer)
        test(model_cnn, valid_loader)
    

