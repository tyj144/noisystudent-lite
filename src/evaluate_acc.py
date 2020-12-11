from __future__ import print_function, division
import sys
import copy
import os
import datetime
import time
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import torchvision
import numpy as np
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch
import os
from shared import data_path, data_split, weights_path
from randaugment import RandAugment

# path of weights transfered from CCV
if len(sys.argv) == 3:
    PATH = weights_path(sys.argv[2])
else:
    print("Please include weights file name")
    exit()

# for compatibility with old code
BATCH_SIZE = 32

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

dataset_name = "CUB_200"
data_dir = data_path(dataset_name)
dataset = datasets.ImageFolder(data_dir, data_transforms['train'])

image_datasets, test_set = data_split(dataset)

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = [classname.split('.')[1].replace('_', ' ') for classname in dataset.classes]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model_loaded = models.resnet50()
num_ftrs = model_loaded.fc.in_features
model_loaded.fc = nn.Linear(num_ftrs, len(class_names))

model_loaded = model_loaded.to(device)
model_loaded.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

def evaluate_model(model, test_set):
    model.eval()
    loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)
    print('Test set size:', len(test_set))
    with torch.no_grad():
        running_corrects = 0
        seen = 0
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels)
            seen += len(inputs)
    return running_corrects / seen

print('Final accuracy:', evaluate_model(model_loaded, test_set))