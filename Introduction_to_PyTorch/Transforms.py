from Datasets_and_DataLoaders import display_image_and_label

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) #one-hotの形にする
)

if __name__ == '__main__':
    train_dataloader = DataLoader(ds, batch_size=64, shuffle=True)
    display_image_and_label(train_dataloader)