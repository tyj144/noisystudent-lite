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

# for compatibility with old code
is_local = True

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
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

# dataset_name = "CUB_200"
dataset_name = "CUB_200_2011/CUB_200_2011/images"
data_dir = f"datasets/{dataset_name}" if is_local else f"/users/tjiang12/data/tjiang12/{dataset_name}"
dataset = datasets.ImageFolder(data_dir, data_transforms['train'])

val_size = int(0.3 * len(dataset))
train_size = len(dataset) - val_size

train_and_val = torch.utils.data.random_split(dataset, [train_size, val_size])

image_datasets = dict(zip(['train', 'val'], train_and_val))
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                              shuffle=True)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = [classname.split('.')[1].replace('_', ' ') for classname in dataset.classes]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# path of weights transfered from CCV
# PATH = "weights/student_alternating_1"
num_classes = 200
PATH = "weights/resnet50_CUB200_66pct"
model_loaded = models.resnet50()
num_ftrs = model_loaded.fc.in_features
model_loaded.fc = nn.Linear(num_ftrs, num_classes)

model_loaded = model_loaded.to(device)
model_loaded.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))


# inputs, labels = dataset.__getitem__(0)
# print(inputs)
# print(torch.reshape(inputs, (1,3,224,224)))
# print(torch.reshape(inputs, (1,3,224,224))[0] == inputs)

# import random
# with torch.no_grad():
#     running_corrects = 0
#     seen = 0
#     # for i, (inputs, labels) in enumerate(dataset):
#     while True:
#         i = random.randint(0, len(dataset))
#         print(i)
#         inputs, labels = dataset.__getitem__(i)
#         inputs = inputs.to(device)

#         outputs = model_loaded(torch.reshape(inputs, (1,3,224,224)))
#         _, preds = torch.max(outputs, 1)
#         print(preds, labels)
#         running_corrects += torch.sum(preds == labels)
#         seen += 1
#         print(running_corrects, "/", seen, running_corrects / seen)

student = models.resnet50(pretrained=True)

# Augment last layer to match dimensions
num_classes = 200
num_ftrs = student.fc.in_features
student.fc = nn.Linear(num_ftrs, num_classes)
student = student.to(device)

# Experiment: Evaluate trained model's performance on unlabeled data
model_loaded.eval()

with torch.no_grad():
    running_corrects = 0
    seen = 0
    for i, (inputs, truelabels) in enumerate(dataloaders['val']):
        inputs = inputs.to(device)

        teacher_out = model_loaded(inputs)
        _, pseudolabels = torch.max(teacher_out, 1)
        student_out = student(inputs)
        _, preds = torch.max(student_out, 1)
        print(preds)
        print("sanity check", pseudolabels[0], truelabels[0])
        running_corrects += torch.sum(preds == pseudolabels)
        seen += len(inputs)
        print(running_corrects, "/", seen, running_corrects / seen)


# from __future__ import print_function, division
# 
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
# import time
# import os
# import copy
# import datetime
# import sys
# from shared import weights_path, data_path, gen_train_val, data_transforms, should_print
# from PseudolabelDataset import PseudolabelDataset

# is_local = len(sys.argv) == 2 and sys.argv[1] == 'local'
# NUM_EPOCHS = 100
# BATCH_SIZE = 32

# # Load smaller labeled set (CUB 200 2010)
# labeled_dataset_name = "CUB_200"
# labeled_data_dir = data_path(labeled_dataset_name)
# labeled_dataset = datasets.ImageFolder(
#     labeled_data_dir, data_transforms['train'])

# labeled_image_datasets = gen_train_val(labeled_dataset)
# labeled_dataloaders = {x: torch.utils.data.DataLoader(labeled_image_datasets[x], batch_size=BATCH_SIZE,
#                                               shuffle=True)
#                for x in ['train', 'val']}
# labeled_dataset_sizes = {x: len(labeled_image_datasets[x]) for x in ['train', 'val']}

# class_names = labeled_dataset.classes

# # Load larger unlabeled set (CUB 200 2011)
# unlabeled_data_dir = data_path("CUB_200_2011/CUB_200_2011/images")
# unlabeled_dataset = datasets.ImageFolder(
#     unlabeled_data_dir, data_transforms['train'])
    

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# run_name = f"student_CUB200_{NUM_EPOCHS}_{str(datetime.datetime.now())}"
# PATH = weights_path(run_name)

# # path of weights transfered from CCV
# teacher_path = "weights/resnet50_CUB200_66pct"

# PATH = "weights/resnet50_CUB200_66pct"
# model_loaded = models.resnet50()
# num_ftrs = model_loaded.fc.in_features
# model_loaded.fc = nn.Linear(num_ftrs, 200)

# model_loaded = model_loaded.to(device)
# model_loaded.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

# # Experiment: Evaluate trained model's performance on unlabeled data
# model_loaded.eval()

# # with torch.no_grad():
# #     running_corrects = 0
# #     seen = 0
# #     for i, (inputs, labels) in enumerate(unlabeled_dataset):
# #         inputs = inputs.to(device)

# #         outputs = model_loaded(torch.reshape(inputs, (1,3,224,224)))
# #         _, preds = torch.max(outputs, 1)
# #         print(preds)
# #         running_corrects += torch.sum(preds == labels)
# #         seen += 1
# #         print(running_corrects, "/", seen, running_corrects / seen)

# # Configure teacher model's architecture and load in pre-trained model
# teacher = models.resnet50()
# # Augment last layer to match dimensions
# num_ftrs = teacher.fc.in_features
# teacher.fc = nn.Linear(num_ftrs, len(class_names))

# teacher = teacher.to(device)
# print(teacher_path)
# # Load in trained model as teacher
# teacher.load_state_dict(torch.load(
#     teacher_path, map_location=torch.device('cpu')))

# softmax = nn.Softmax(dim=1)
# # for i, j in unlabeled_dataset:
# #     logits = teacher(torch.reshape(i, (1, 3, 224, 224)))
# #     probs = softmax(logits)
# #     value, prediction = torch.max(probs, dim=1)
# #     print('prediction', prediction, value)
# #     pseudolabel = int(prediction)

# # Generate dataloaders for the labeled and pseudolabeled together
# pseudo_dataset = PseudolabelDataset(unlabeled_dataset, teacher=teacher, device=device)
# pseudo_image_datasets = gen_train_val(pseudo_dataset)
# pseudo_dataloaders = {x: torch.utils.data.DataLoader(pseudo_image_datasets[x], batch_size=BATCH_SIZE,
#                                               shuffle=True)
#                for x in ['train', 'val']}
# pseudo_dataset_sizes = {x: len(pseudo_image_datasets[x]) for x in ['train', 'val']}

# for i, j in pseudo_dataset:
#     print(j)

# # for i, j in pseudo_dataloaders['train']:
# #     print(j)