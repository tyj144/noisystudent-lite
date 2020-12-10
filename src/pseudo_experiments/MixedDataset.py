import torch
import torch.nn as nn
from PseudolabelDataset import PseudolabelDataset

softmax = nn.Softmax(dim=1)

class MixedDataset(torch.utils.data.Dataset):
    '''
    A dataset that includes both the pseudo-labeled and correctly labeled data.
    '''

    def __init__(self, labeled, unlabeled, teacher=None, threshold=0.75, transform=None, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.labeled = labeled
        self.device = device
        # teacher model for pseudo-labeling
        self.pseudolabeled = PseudolabelDataset(unlabeled, teacher=teacher, threshold=threshold, device=device) \
            if teacher is not None else None

    def __len__(self):
        if self.pseudolabeled is None:
            return len(self.labeled)
        return len(self.labeled) + len(self.pseudolabeled)

    def __getitem__(self, idx):
        if idx < len(self.labeled):
            img, label = self.labeled.__getitem__(idx)
            return img.to(self.device), label

        if self.pseudolabeled is not None:
            idx = idx - len(self.labeled)
            img, pseudolabel = self.pseudolabeled.__getitem__(idx)
            print(pseudolabel)
            return img, pseudolabel