# Train a ResNet-50 pre-trained on ImageNet on CUB200.
from __future__ import print_function, division
import sys
import copy
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

from shared import data_split, data_path, weights_path, should_print, data_transforms

DATASET_NAME = "CUB_200"
NUM_EPOCHS = 100
BATCH_SIZE = 16

plt.ion()   # interactive mode

# Fetch the path to the dataset
data_dir = data_path(DATASET_NAME)
# Noise the labeled data
dataset = datasets.ImageFolder(data_dir, data_transforms['train'])
# Fetch the datasets with train-val split of 0.7 and 0.3
image_datasets, test_set = data_split(dataset)

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                              shuffle=True)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = dataset.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Where to save the model
WEIGHTS_PATH = weights_path(f"resnet50_{DATASET_NAME}_{NUM_EPOCHS}_{str(datetime.datetime.now())}")


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
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
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

                with torch.set_grad_enabled(phase == 'train'):
                    # forward pass
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(running_corrects.double(), "/", dataset_sizes[phase])
            print(running_corrects.double() / dataset_sizes[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                print("UPDATE:", best_acc, "to", epoch_acc)
                print("Saving model to", WEIGHTS_PATH)
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), WEIGHTS_PATH)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

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
            if should_print(i):
                print(preds)
                print(int(running_corrects), "/", seen, running_corrects / seen)
    return running_corrects / seen


model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=NUM_EPOCHS)
                       
accuracy = evaluate_model(model_ft, test_set)