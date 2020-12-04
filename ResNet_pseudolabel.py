from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import datetime
import sys

is_local = len(sys.argv) == 2 and sys.argv[1] == 'local'
NUM_EPOCHS = 100

# path of weights transfered from CCV
TEACHER_NAME = "resnet50_CUB200_66pct"
TRAINED_MODEL_PATH = f"weights/{TEACHER_NAME}" if is_local else f"/users/tjiang12/scratch/{TEACHER_NAME}"

# Transforms to apply to each image
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

# Load smaller labeled set (CUB 200 2010)
labeled_dataset_name = "CUB_200"
labeled_data_dir = f"datasets/{labeled_dataset_name}" if is_local else f"/users/tjiang12/data/tjiang12/{labeled_dataset_name}"
labeled_dataset = datasets.ImageFolder(
    labeled_data_dir, data_transforms['train'])

class_names = labeled_dataset.classes

# Load larger unlabeled set (CUB 200 2011)
unlabeled_dataset_name = "CUB_200_2011/CUB_200_2011/images"
unlabeled_data_dir = f"datasets/{unlabeled_dataset_name}" if is_local else f"/users/tjiang12/data/tjiang12/{unlabeled_dataset_name}"
unlabeled_dataset = datasets.ImageFolder(
    unlabeled_data_dir, data_transforms['train'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RUN_NAME = f"student_CUB200_{NUM_EPOCHS}_{str(datetime.datetime.now())}"
PATH = f"weights/{RUN_NAME}" if is_local else f"/users/tjiang12/scratch/{RUN_NAME}"

# Configure teacher model's architecture
teacher = models.resnet50()

# Augment last layer to match dimensions
num_classes = 200
num_ftrs = teacher.fc.in_features
teacher.fc = nn.Linear(num_ftrs, num_classes)

teacher = teacher.to(device)

# Load in trained model as teacher
teacher.load_state_dict(torch.load(
    TRAINED_MODEL_PATH, map_location=torch.device('cpu')))

sm = nn.Softmax(dim=1)

# Write a dataloader that randomly picks either the pseudo labeled or correctly labeled data


class StudentDataset(torch.utils.data.Dataset):

    def __init__(self, teacher, labeled, unlabeled, transform=None):
        self.teacher = teacher
        self.teacher = self.teacher.to(device)
        self.labeled = labeled
        self.unlabeled = unlabeled

    def __len__(self):
        return len(self.labeled)
        # return len(self.labeled) + len(self.unlabeled)

    def __getitem__(self, idx):
        if idx < len(self.labeled):
            img, label = self.labeled.__getitem__(idx)
            return img.to(device), label

        # idx = idx - len(self.labeled)
        # img, _ = self.unlabeled.__getitem__(idx)
        # img = img.to(device)
        # logits = self.teacher(torch.reshape(img, (1, 3, 224, 224)))
        # probs = sm(logits)
        # value, prediction = torch.max(probs, dim=1)
        # pseudolabel = int(prediction)
        # if value < 0.75:
        #     return img, -1
        # return img, pseudolabel


combined_dataset = StudentDataset(teacher, labeled_dataset, unlabeled_dataset)

val_size = int(0.3 * len(combined_dataset))
train_size = len(combined_dataset) - val_size

train_and_val = torch.utils.data.random_split(
    combined_dataset, [train_size, val_size])

image_datasets = dict(zip(['train', 'val'], train_and_val))
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                              shuffle=True)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(dataset_sizes)

# Configure student model's architecture
student = models.resnet50(pretrained=True)

# Augment last layer to match dimensions
num_classes = 200
num_ftrs = student.fc.in_features
student.fc = nn.Linear(num_ftrs, num_classes)

student = student.to(device)


def should_print(i):
    print_every = 15 if is_local else 200
    return i % print_every == 0


sm = nn.Softmax(dim=1)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # track best model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # train for num_epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a phase: train and val
        for phase in ['train', 'val']:
            epoch_begin = time.time()
            if phase == 'train':
                # sets it in training mode
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # get batch
            num_noconf_labels = 0
            examples_used = 0
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                valid = labels != -1
                if torch.sum(valid) <= 0:
                    num_noconf_labels += 1
                    continue
                # print(valid)
                # print(inputs.shape)
                inputs = inputs[valid]
                labels = labels[valid]
                examples_used += torch.sum(valid)
                # print(torch.sum(valid))
                # print(inputs.shape)
                # print(labels.shape)
                # print(labels)
                if should_print(i):
                    time_elapsed = time.time() - epoch_begin
                    print(
                        i + 1, '/', len(dataloaders[phase]), int(time_elapsed), 'seconds')
                    print('ETA:', datetime.timedelta(seconds=int(
                        (time_elapsed / (i + 1)) * (len(dataloaders[phase]) - i))))
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero out the previous gradient
                optimizer.zero_grad()

                # dunno what this `with` does
                with torch.set_grad_enabled(phase == 'train'):
                    # forward pass
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    if should_print(i):
                        print(preds[:10], labels[:10], torch.sum(
                            preds[:10] == labels[:10]), "out of", 10)
                        print(preds.shape, labels.shape)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            print("Number of full -1 batches:", num_noconf_labels,
                  "out of", int(len(dataloaders[phase]) / 8))
            print("Number of examples used (enough conf.):", examples_used,
                  "out of", len(dataloaders[phase]) * 8)

            epoch_loss = running_loss / examples_used
            epoch_acc = running_corrects.double() / examples_used

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                print("UPDATE:", best_acc, "to", epoch_acc)
                print("Saving model to", PATH)
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), PATH)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(student.parameters(), lr=0.01, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

student = train_model(student, criterion, optimizer_ft, exp_lr_scheduler,
                      num_epochs=NUM_EPOCHS)
