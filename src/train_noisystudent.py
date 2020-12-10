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
from shared import weights_path, data_path, data_split, gen_train_val, data_transforms, should_print
from randaugment import RandAugment

is_local = len(sys.argv) == 2 and sys.argv[1] == 'local'
NUM_EPOCHS = 100
BATCH_SIZE = 32

# Load smaller labeled set (CUB 200 2010)
labeled_dataset_name = "CUB_200"
labeled_data_dir = data_path(labeled_dataset_name)
# Noise the labeled data
labeled_dataset = datasets.ImageFolder(
    labeled_data_dir, transforms.Compose([
    RandAugment(),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),                        
]))

labeled_image_datasets, test_set = data_split(labeled_dataset)
labeled_dataloaders = {x: torch.utils.data.DataLoader(labeled_image_datasets[x], batch_size=BATCH_SIZE,
                                              shuffle=True)
               for x in ['train', 'val']}
labeled_dataset_sizes = {x: len(labeled_image_datasets[x]) for x in ['train', 'val']}

class_names = labeled_dataset.classes

# Load larger unlabeled set (CUB 200 2011)
unlabeled_data_dir = data_path("CUB_200_2011/CUB_200_2011/images")
unlabeled_dataset = datasets.ImageFolder(
    unlabeled_data_dir, data_transforms['train'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

run_name = f"noisystudent_CUB200_{NUM_EPOCHS}_{str(datetime.datetime.now())}".replace(" ", "_")
PATH = weights_path(run_name)

# path of weights transfered from CCV
teacher_path = weights_path("resnet50_CUB200_66pct")

# Configure teacher model's architecture and load in pre-trained model
teacher = models.resnet50()
# Augment last layer to match dimensions
num_ftrs = teacher.fc.in_features
teacher.fc = nn.Linear(num_ftrs, len(class_names))

teacher = teacher.to(device)
print(teacher_path)
# Load in trained model as teacher
teacher.load_state_dict(torch.load(
    teacher_path, map_location=torch.device('cpu')))

# Generate dataloaders for the labeled and pseudolabeled together
# pseudo_dataset = PseudolabelDataset(unlabeled_dataset, teacher=teacher, device=device)
unlabeled_image_datasets = gen_train_val(unlabeled_dataset)
unlabeled_dataloaders = {x: torch.utils.data.DataLoader(unlabeled_image_datasets[x], batch_size=BATCH_SIZE,
                                              shuffle=True)
               for x in ['train', 'val']}
c_dataset_sizes = {x: len(unlabeled_image_datasets[x]) for x in ['train', 'val']}
# print(dataset_sizes)

# Configure student model's architecture
student = models.resnet50(pretrained=True)

# Augment last layer to match dimensions
num_classes = 200
num_ftrs = student.fc.in_features
student.fc = nn.Linear(num_ftrs, num_classes)
student = student.to(device)

f = open(run_name, "w")

def print_write(*s):
    print(*s, file=f)
    print(*s)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # track best model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # train for num_epochs
    for epoch in range(num_epochs):
        train_on_pseudo = epoch < num_epochs / 2
        print_write('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print_write('-' * 10)

        if train_on_pseudo:
            print_write("Psuedolabeled run")
            dataloaders = unlabeled_dataloaders
        else:
            print_write("Labeled run")
            dataloaders = labeled_dataloaders

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
            # num_noconf_labels = 0
            examples_used = 0
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                # valid = labels != -1
                # if torch.sum(valid) <= 0:
                #     num_noconf_labels += 1
                #     continue
                # # print_write(valid)
                # # print_write(inputs.shape)
                # inputs = inputs[valid]
                # labels = labels[valid]
                # examples_used += torch.sum(valid)
                examples_used += len(inputs)
                # print_write(torch.sum(valid))
                # print_write(inputs.shape)
                # print_write(labels.shape)
                # print_write(labels)
                if should_print(i):
                    time_elapsed = time.time() - epoch_begin
                    print_write(
                        i + 1, '/', len(dataloaders[phase]), int(time_elapsed), 'seconds')
                    print_write('ETA:', datetime.timedelta(seconds=int(
                        (time_elapsed / (i + 1)) * (len(dataloaders[phase]) - i))))

                # generate pseudolabels
                inputs = inputs.to(device)
                labels = labels.to(device)
                if train_on_pseudo:
                    teacher_out = teacher(inputs)
                    _, pseudolabels = torch.max(teacher_out, 1)
                    if should_print(i):
                        print("Pseudolabel acc.:", torch.sum(labels == pseudolabels) / len(labels))
                    labels = pseudolabels


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
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            # print("Number of full -1 batches:", num_noconf_labels,
            #       "out of", int(len(dataloaders[phase]) / BATCH_SIZE))
            # print("Number of examples used (enough conf.):", examples_used,
            #       "out of", len(dataloaders[phase]) * BATCH_SIZE)

            epoch_loss = running_loss / examples_used
            epoch_acc = running_corrects.double() / examples_used

            print_write('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model (only on noisily labeled)
            if not train_on_pseudo and phase == 'val' and epoch_acc > best_acc:
                print_write("UPDATE:", best_acc, "to", epoch_acc)
                print_write("Saving model to", PATH)
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), PATH)

        print_write()

    time_elapsed = time.time() - since
    print_write('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print_write('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, test_set):
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


criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(student.parameters(), lr=0.01, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

student = train_model(student, criterion, optimizer_ft, exp_lr_scheduler,
                      num_epochs=NUM_EPOCHS)
accuracy = evaluate_model(student, test_set)
