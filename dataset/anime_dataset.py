import os
import sys
import h5py 

import numpy as np
import torch
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from torchvision.utils import save_image


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Anime_Dataset_H5(Dataset):
    def __init__(self, root, mode, transform=None):
        assert mode in ["year", "hair_eye", "hair", "eye"]
        self.mode = mode
        self.root = root
        self.transform = transform

        with h5py.File(root, 'r') as hf:
            if self.mode == "year":
                self.imgs = hf["year_imgs"][...]
                self.labels = hf["year_labels"][...]
                self.num_classes = np.max(self.labels) + 1 # 0 - l (l + 1 class)
            elif self.mode == "hair_eye":
                self.imgs = hf["hair_eye_imgs"][...]
                self.labels = hf["hair_eye_labels"][...]
                self.num_classes = None
            assert self.imgs.shape[0] == self.labels.shape[0]

    def __len__(self):
        return self.imgs.shape[0]
    
    def __getitem__(self, i):
        img, label = self.imgs[i], self.labels[i]
        if self.transform is not None:
            img = self.transform(img)
        label = F.one_hot(torch.tensor(label), self.num_classes)
        return img, label


        


class Anime_Dataset:
    def __init__(self, root, class_num, transform):
        self.root = root
        self.img_folder = os.path.join(self.root, 'images')
        self.label_file = os.path.join(self.root, 'labels.pkl')
        self.img_files = os.listdir(self.img_folder)
        self.labels = pickle.load(open(self.label_file, 'rb'))
        self.preprocess()
        self.class_num = class_num
        self.transform = transform
        
        assert(len(self.img_files) <= len(self.labels))
    
    def preprocess(self):
        new_label = {}
        for img, tag in self.labels.items():
            if tag[-1] is None:
                new_label[img] = tag[:-1]
        self.labels = new_label
        self.img_files = [path for path in self.img_files if os.path.splitext(path)[0] in self.labels]
        print(len(self.labels), len(self.img_files))
    
    def color_transform(self, x):
        x = TF.adjust_saturation(x, 2.5)
        x = TF.adjust_gamma(x, 0.7)
        x = TF.adjust_contrast(x, 1.2)
        return x
        
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_folder, self.img_files[idx]))
        img = self.color_transform(img)
        img = self.transform(img)
        filename = os.path.splitext(self.img_files[idx])[0]
        label = self.labels[filename]
        
        one_hots = []
        mask = []
        for i, c in enumerate(self.class_num):
            l = torch.zeros(c)
            m = torch.zeros(c)
            if label[i]:
                l[label[i]] = 1
                m = 1 - m # create mask
            one_hots.append(l)
            mask.append(m)
        one_hots = torch.cat(one_hots, 0)
        mask = torch.cat(mask, 0)
        return img, one_hots, mask

def get_anime_dataloader(root, classes, batch_size):
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = Anime_Dataset(root, classes, transform)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)
    return dataloader

def get_anime_h5_dataloader(root, batch_size):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(0.0, 0.2, 0.2, 0.0), # color jitter
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # [0, 1] to [-1, 1]
        transforms.RandomHorizontalFlip(p = 0.5)
    ])
    dataset = Anime_Dataset_H5(root, "year", transform)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True, num_workers=4)
    return dataloader
    
def denorm(img):
    """ Denormalize input image tensor. (From [0,1] -> [-1,1]) 
    
    Args:
        img: input image tensor.
    """
	
    output = img / 2 + 0.5
    return output.clamp(0, 1)

if __name__ == '__main__':
    dataloader = get_anime_h5_dataloader('/mnt/data2/bchao/anime/data/dataset.h5', 50)
    img, label = next(iter(dataloader))
    save_image(denorm(img), "test.png")
    