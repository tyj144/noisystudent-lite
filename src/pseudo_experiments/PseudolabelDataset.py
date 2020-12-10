import torch
import torch.nn as nn

softmax = nn.Softmax(dim=1)

class PseudolabelDataset(torch.utils.data.Dataset):
    def __init__(self, data, teacher, threshold=0, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        dataloader = torch.utils.data.DataLoader(data, batch_size=1,
                                              shuffle=True)
        self.dataiter = iter(dataloader)
        self.teacher = teacher
        self.threshold = threshold
        self.device = device

    def __len__(self):
        return len(self.dataiter)

    def __getitem__(self, idx):
        print("asdf", idx)
        img, label = next(self.dataiter)
        img = img.to(self.device)

        logits = self.teacher(img)
        probs = softmax(logits)
        value, prediction = torch.max(probs, dim=1)
        print('prediction', prediction, value, label)
        pseudolabel = int(prediction)
        if value < self.threshold:
            return img[0], -1
        return img[0], pseudolabel
