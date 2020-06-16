import os
import sys
import torch
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.utils import save_image


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
        x = F.adjust_saturation(x, 2.5)
        x = F.adjust_gamma(x, 0.7)
        x = F.adjust_contrast(x, 1.2)
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
    
def denorm(img):
    """ Denormalize input image tensor. (From [0,1] -> [-1,1]) 
    
    Args:
        img: input image tensor.
    """
	
    output = img / 2 + 0.5
    return output.clamp(0, 1)

if __name__ == '__main__':
    dataloader = get_anime_dataloader('../../data', (10, 12), 50)
    
    img, label, mask = next(iter(dataloader))
    print(img.shape)
    print(label.shape)
    print(mask)
    save_image(denorm(img), 'test.png')
    