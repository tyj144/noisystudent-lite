import torch
from torchvision import datasets, transforms
import torch.nn as nn
import os
import sys

is_local = len(sys.argv) == 2 and sys.argv[1] == 'local'

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

def get_dataset(data_dir, val_pct=0.3):
    '''
    Take in a path to a dataset, get the datasets with train/val split and classnames.
    '''
    dataset = datasets.ImageFolder(data_dir, data_transforms['train'])
    image_datasets = gen_train_val(dataset, val_pct=val_pct)
    return image_datasets, dataset.classes

def gen_train_val(dataset, val_pct=0.3):
    '''
    Generate the train/val split from a dataset.
    '''
    val_size = int(val_pct * len(dataset))
    train_size = len(dataset) - val_size

    train_and_val = torch.utils.data.random_split(dataset, [train_size, val_size])

    image_datasets = dict(zip(['train', 'val'], train_and_val))
    return image_datasets

def data_split(dataset, val_pct=0.1, test_pct=0.2):
    '''
    Generate the train/val/test split from a dataset.
    '''
    val_size = int(val_pct * len(dataset))
    test_size = int(test_pct * len(dataset))
    train_size = len(dataset) - val_size - test_size

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    image_datasets = { 'train': train_set, 'val': val_set }
    return image_datasets, test_set

def data_path(dataset_name):
    prefix = "~/dev/noisystudent-lite/datasets/" if is_local else "/users/tjiang12/data/tjiang12/"
    return os.path.join(prefix, dataset_name)

def weights_path(run_name):
    prefix = f"weights/" if is_local else f"/users/tjiang12/scratch/"
    return os.path.join(prefix, run_name)

def should_print(i):
    '''
    Manage printing during training (i.e. ETA calculations.)
    '''
    print_every = 15 if is_local else 200
    return i % print_every == 0
