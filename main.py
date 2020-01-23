import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
import os
import argparse

from model import CNN1, CNN2
from dataset import MyDataset
from trainer import train, test, matplotlib_imshow
from pathlib import Path
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


def main():
    # Add argument parser 
    parser = argparse.ArgumentParser(description='Dog vs Cat Example')
    parser.add_argument('--data', 
                        type=str, default='/home/aims/Dropbox/AMMI/Tutorial/NN_1/project/Cat_Dog_data', 
                        help='Folder that contain your training and testing data')
    args = parser.parse_args()
    writer = SummaryWriter('runs/catvsdog_experiment_1')
    
    # set path
    PATH=Path(args.data)
    TRAIN =Path(PATH/'train')
    VALID = Path(PATH/'test')

    # set hyperparameter
    num_workers = 0
    batch_size = 32
    n_features = 6
    input_size = 224
    output_size = 2

    # create custom tranform
    viz_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor()])
    train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    # Loading the data using custom data loader
    viz_data = MyDataset(TRAIN, transform=viz_transforms)
    train_data = MyDataset(TRAIN, transform=train_transforms)
    valid_data = MyDataset(VALID,transform=valid_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,  num_workers=num_workers,shuffle=True)
    
    viz_loader = torch.utils.data.DataLoader(viz_data, batch_size=batch_size, num_workers=num_workers,shuffle=True)

    # Create model
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model_cnn = CNN2(input_size, n_features, output_size)
    optimizer = optim.Adam(model_cnn.parameters(), lr=0.01)
    
    # get some random training images
    dataiter = iter(viz_loader)
    images, labels = dataiter.next()

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    matplotlib_imshow(img_grid)

    # write to tensorboard
    writer.add_image('four_dogvscats', img_grid)
    
    writer.add_graph(model_cnn, images)

    # get the class labels for each image
    classes  = ('cat', 'dog')
    class_labels = [classes[lab] for lab in labels]

    # log embeddings
    features = images.view(-1, 224 * 224 * 3)
    writer.add_embedding(features,
                         metadata=class_labels,
                         label_img=images)
    
    model_cnn = model_cnn.to(device)

    for epoch in range(20):
        train(epoch, model_cnn, train_loader, optimizer, writer, device)
        test(model_cnn, valid_loader, device)
    writer.close()

if __name__ == '__main__':
    main()