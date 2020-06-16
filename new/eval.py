import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils.utils import load_model, denorm
import copy

from models.ACGAN import Generator, Discriminator

# 10, 12, 9

epoch = 28700
class_num = (10, 12)
noise_dim = 100

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

def sample_class_fix(batch_size):
    labels = []
        
    for c in class_num:
        label = torch.LongTensor(batch_size, 1).random_() % c
        one_hot = torch.zeros(batch_size, c).scatter(1, label, 1)
        labels.append(one_hot)
            
    labels = torch.cat(labels, 1)
    return labels
    
def generate_class_gradient(class_num, change_dim):
    G = load_G(G_path).cuda()
    
    c = sample_class_gradient(class_num, change_dim).cuda()
    z = torch.empty(1, 100).normal_(0, 0.9).repeat(class_num[change_dim], 1).cuda()
    
    img = denorm(G(z, c))
    save_image(img, 'test.png', nrow = class_num[change_dim])

def generate_class_fix(batch_size, best_size = 64): 
    G = load_G(G_path).cuda()
    D = load_D(D_path).cuda()
    
    torch.set_printoptions(threshold = 5000)
    c = sample_class_fix(1).repeat(batch_size, 1).cuda()
    z = torch.empty(batch_size, noise_dim).normal_(0, 0.9).cuda()
    
    img = G(z, c)
    score, pred = D(img)
    print(score)
    top_score, idx = torch.topk(score, best_size)
    idx = idx.cpu().detach().numpy()
    print(idx, top_score)
    
    img = denorm(img[idx])
    save_image(img, 'test.png')
    #save_image(top_score, 'score.png')
    
    
#generate_class_gradient(class_num, 0)
generate_class_fix(1024)





