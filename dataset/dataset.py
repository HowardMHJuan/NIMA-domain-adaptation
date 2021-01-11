"""
file - dataset.py
Customized dataset class to loop through the AVA dataset and apply needed image augmentations for training.

Copyright (C) Yunxiao Shi 2017 - 2020
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""

import os

import pandas as pd
from PIL import Image

import torch
from torch.utils import data
import torchvision.transforms as transforms


class AVADataset(data.Dataset):
    """AVA dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name).convert('RGB')
        annotations = self.annotations.iloc[idx, 1:].to_numpy()
        annotations = annotations.astype('float').reshape(-1, 1)
        label = 0
        for score, prob in enumerate(annotations):
            label += score * prob
        label = label.round().item()
        sample = {'img_id': img_name, 'image': image, 'annotations': annotations, 'label': label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


class MEDataset(data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.label_df = pd.read_csv(csv_file, delimiter="\t", header=None)
        self.root_dir = root_dir
        self.transform = transform

        self.annotation_list = torch.zeros((len(self.label_df), 10))
        for idx, label in self.label_df.iterrows():
            for l in label:
                self.annotation_list[idx, l] += 1
        self.annotation_list = self.annotation_list / 3

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f'{idx + 1}.png')
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        annotations = self.annotation_list[idx]
        annotations = annotations.type(torch.float32).view(-1, 1)

        label = 0
        for score, prob in enumerate(annotations):
            label += score * prob
        label = label.round().item()

        return {'image': image, 'annotations': annotations, 'label': label}


class GroupDataset(data.Dataset):
    def __init__(self, s_loader, t_loader, t_size):
        self.s_loader = s_loader
        self.s_loader_iter = iter(s_loader)
        self.t_batch = next(iter(t_loader))
        self.t_size = t_size
        self.shot = t_size // 10
    
    def _get_next_s_batch(self):
        while True:
            s_batch = next(self.s_loader_iter)
            if (
                torch.any(s_batch["label"] == 7)
                and torch.any(s_batch["label"] == 6)
                and torch.any(s_batch["label"] == 3)
                and torch.any(s_batch["label"] == 2)
            ):
                return s_batch

    def update_and_process(self):
        try:
            s_batch = self._get_next_s_batch()
        except StopIteration:
            self.s_loader_iter = iter(self.s_loader)
            s_batch = self._get_next_s_batch()

        classes = torch.Tensor(list(range(2, 7+1)))
        classes = classes[torch.randperm(len(classes))]

        def s_get_cls_shots(c):
            idxs = torch.nonzero(s_batch["label"].eq(c)).view(-1)
            idxs = idxs[torch.randperm(len(idxs))]
            while len(idxs) < self.shot * 2:
                idxs = torch.cat([idxs, idxs])
            return idxs[:self.shot * 2]
        def t_get_cls_shots(c):
            idxs = torch.nonzero(self.t_batch["label"].eq(c)).view(-1)
            idxs = idxs[torch.randperm(len(idxs))]
            while len(idxs) < self.shot:
                idxs = torch.cat([idxs, idxs])
            return idxs[:self.shot]

        s_idxs = [s_get_cls_shots(c) for c in classes]
        t_idxs = [t_get_cls_shots(c) for c in classes]

        s_idxs = torch.stack(s_idxs)
        t_idxs = torch.stack(t_idxs)

        self.G1, self.G2, self.G3, self.G4 = [], [] ,[] ,[]
        self.Y1, self.Y2, self.Y3, self.Y4 = [], [] ,[] ,[]

        for i in range(6):
            for j in range(self.shot):
                self.G1.append((s_batch["image"][s_idxs[i][j*2]], s_batch["image"][s_idxs[i][j*2+1]]))
                self.Y1.append((s_batch["annotations"][s_idxs[i][j*2]], s_batch["annotations"][s_idxs[i][j*2+1]]))
                self.G2.append((s_batch["image"][s_idxs[i][j]], self.t_batch["image"][t_idxs[i][j]]))
                self.Y2.append((s_batch["annotations"][s_idxs[i][j]], self.t_batch["annotations"][t_idxs[i][j]]))
                self.G3.append((s_batch["image"][s_idxs[i][j]], s_batch["image"][s_idxs[(i+1)%6][j]]))
                self.Y3.append((s_batch["annotations"][s_idxs[i][j]], s_batch["annotations"][s_idxs[(i+1)%6][j]]))
                self.G4.append((s_batch["image"][s_idxs[i][j]], self.t_batch["image"][t_idxs[(i+1)%6][j]]))
                self.Y4.append((s_batch["annotations"][s_idxs[i][j]], self.t_batch["annotations"][t_idxs[(i+1)%6][j]]))

    def __len__(self):
        return self.shot * 6

    def __getitem__(self, idx):
        return {
            "G1": self.G1[idx],
            "Y1": self.Y1[idx],
            "G2": self.G2[idx],
            "Y2": self.Y2[idx],
            "G3": self.G3[idx],
            "Y3": self.Y3[idx],
            "G4": self.G4[idx],
            "Y4": self.Y4[idx],
        }

if __name__ == '__main__':
    # sanity check
    transform = transforms.Compose([
        transforms.Scale(256), 
        transforms.RandomCrop(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor()
    ])

    ava_set = AVADataset(csv_file="data/AVA/train_label.csv", root_dir="/tmp/r09922083/AVA/images", transform=transform)
    ava_loader = data.DataLoader(ava_set, batch_size=2, shuffle=True, num_workers=8)
    for batch in ava_loader:
        print(batch["label"])
        print(batch["image"].size(), batch["annotations"].size())
        break

    me_set = MEDataset(csv_file="data/ME/label.csv", root_dir="data/ME/images", transform=transform)
    me_loader = data.DataLoader(me_set, batch_size=2, shuffle=True, num_workers=8)
    for batch in me_loader:
        print(batch["label"])
        print(batch["image"].size(), batch["annotations"].size())
        break
    
    s_loader = data.DataLoader(ava_set, batch_size=len(me_set), shuffle=True, num_workers=16)
    t_loader = data.DataLoader(me_set, batch_size=len(me_set), shuffle=True, num_workers=16)

    group_set = GroupDataset(s_loader, t_loader, len(me_set))
    group_set.update_and_process()
    group_loader = data.DataLoader(group_set, batch_size=2, shuffle=True, num_workers=16)
    for batch in group_loader:
        group_list = [
            batch["G1"],
            batch["G2"],
            batch["G3"],
            batch["G4"],
        ]
        y_list = [
            batch["Y1"],
            batch["Y2"],
            batch["Y3"],
            batch["Y4"],
        ]
        print(group_list[0], y_list)
        break
