import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils.utils import load_model, denorm
import copy
import json
import os

from models.ACGAN import Generator, Discriminator

# 10, 12, 9

epoch = 40180
noise_dim = 100
result_dir = 'results'

G_path = 'runs/acgan/ckpt/G_{}.pth'.format(epoch)
D_path = 'runs/acgan/ckpt/D_{}.pth'.format(epoch)

def load_G(G_path):
    prev_state = torch.load(G_path)

    G = Generator(noise_dim, sum(class_num))
    G.load_state_dict(prev_state['model'])
    G.eval()
    return G

def load_D(D_path):
    prev_state = torch.load(D_path)

    D = Discriminator(sum(class_num))
    D.load_state_dict(prev_state['model'])
    D.eval()
    return D

def sample_class_gradient(class_num, change_dim):
        
    labels = []
    for i, c in enumerate(class_num):
        one_hot = torch.zeros(1, c)
        if i != change_dim:
            label = np.random.randint(0, c)
            one_hot[0, label] = 1
        labels.append(one_hot)
    
    batch_labels = [copy.deepcopy(labels) for _ in range(class_num[change_dim])]
    
    for j in range(class_num[change_dim]):
        batch_labels[j][change_dim][0, j] = 1
    
    batch_labels = [torch.cat(label, 1) for label in batch_labels]
    batch_labels = torch.cat(batch_labels, 0)
        
    return batch_labels

def sample_class_fix(class_num, batch_size, fix):
    
    if fix is not None:
        assert(len(fix) == len(class_num))
        assert(all([fix[i] < class_num[i] for i in range(len(class_num))]))
    
    labels = []
    for i, c in enumerate(class_num):
        if fix:
            label = torch.LongTensor(batch_size, 1).fill_(fix[i])
        else:
            label = torch.LongTensor(batch_size, 1).random_() % c
        one_hot = torch.zeros(batch_size, c).scatter(1, label, 1)
        labels.append(one_hot)
            
    labels = torch.cat(labels, 1)
    return labels
    
def generate_class_gradient(class_num, change_dim):
    G = load_G(G_path).cuda()
    dim = 'eye' if change_dim == 0 else 'hair'
    
    c = sample_class_gradient(class_num, change_dim).cuda()
    z = torch.empty(1, 100).normal_(0, 0.9).repeat(class_num[change_dim], 1).cuda()
    
    img = denorm(G(z, c))
    save_image(img, os.path.join(result_dir, 'change_{}.png'.format(dim)), nrow = class_num[change_dim], pad_value = 255)

def generate_class_fix(class_num, batch_size, best_size = 64, fix = None):
    
    eye_label, hair_label = get_label_map()
    print(eye_label, hair_label)
    if fix:
        eye, hair = fix
        img_name = '{}_eye_{}_hair.png'.format(eye_label[eye], hair_label[hair])
    else:
        img_name = 'class_fix.png'
    
    G = load_G(G_path).cuda()
    D = load_D(D_path).cuda()
    
    torch.set_printoptions(threshold = 5000)
    c = sample_class_fix(class_num, 1, fix).repeat(batch_size, 1).cuda()
    z = torch.empty(batch_size, noise_dim).normal_(0, 0.9).cuda()
    
    img = G(z, c)
    score, pred = D(img)
    print(score)
    top_score, idx = torch.topk(score, best_size)
    idx = idx.cpu().detach().numpy()
    print(idx, top_score)
    
    img = denorm(img[idx])
    save_image(img, os.path.join(result_dir, img_name), pad_value = 255)

def generate_class_map(class_num):
    G = load_G(G_path).cuda()
    eye_class, hair_class = class_num
    
    label_batch = []
    for i in range(eye_class):
        eye_label = torch.zeros(1, eye_class)
        eye_label[0, i] = 1
        for j in range(hair_class):
            hair_label = torch.zeros(1, hair_class)
            hair_label[0, j] = 1
            label = torch.cat([eye_label, hair_label], 1)
            label_batch.append(label)
    c = torch.cat(label_batch, 0).cuda()
    z = torch.empty(1, noise_dim).normal_(0, 0.9).repeat(eye_class * hair_class, 1).cuda()
    img = G(z, c)
    
    save_image(denorm(img), os.path.join(result_dir, 'class_map.png'), nrow = hair_class, pad_value = 255)
    
def interpolate(class_num, steps):
    """
        Interpolate.
    """
    G = load_G(G_path).cuda()
    
    z_batch = []
    c_batch = []
    
    c1 = sample_class_fix(class_num, 1, None).cuda()
    c2 = sample_class_fix(class_num, 1, None).cuda()
    z1 = torch.empty(1, noise_dim).normal_(0, 0.9).cuda()
    z2 = torch.empty(1, noise_dim).normal_(0, 0.9).cuda()
    
    c_delta = (c2 - c1) / steps
    z_delta = (z2 - z1) / steps
    for i in range(steps + 1):
        c_batch.append(c1 + i * c_delta)
        z_batch.append(z1 + i * z_delta)
    c = torch.cat(c_batch, 0)
    z = torch.cat(z_batch, 0)
    
    img = G(z, c)
    save_image(denorm(img), os.path.join(result_dir, 'interpolate.png'), nrow = (steps + 1), pad_value = 255)
    
def get_label_map():
    eye_label = json.load(open('data/eye_label.json'))
    hair_label = json.load(open('data/hair_label.json'))
    eye_label = { v: k for k, v in eye_label.items() }
    hair_label = { v: k for k, v in hair_label.items() }
    return eye_label, hair_label
    
if __name__ == '__main__':
    class_num = (10, 12)
    #generate_class_map(class_num)
    generate_class_gradient(class_num, 1)
    #generate_class_fix(class_num, 1024, best_size = 8, fix = (7, 8))
    #interpolate(class_num, 8)
    #get_label_map()




