# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 08:16:32 2018

@author: USER
"""

import torch
import torch.nn
import torch.optim as optim
import torchvision.transforms as Transform
from torchvision.utils import save_image

import numpy as np
import os

import datasets
import ACGAN_2
import utils

hair_mapping =  ['orange', 'white', 'aqua', 'gray', 'green', 'red', 'purple', 
                 'pink', 'blue', 'black', 'brown', 'blonde']

eye_mapping = ['black', 'orange', 'pink', 'yellow', 'aqua', 'purple', 'green', 
               'brown', 'red', 'blue']
               
latent_dim = 100
hair_classes = 12
eye_classes = 10
batch_size = 1

device = 'cpu'

G_path = '../checkpoints/gan_type-[ACGAN]-batch_size-[128]-steps-[50000]-/G_50000.ckpt'


G = G = ACGAN_2.Generator(latent_dim = latent_dim, 
                        code_dim = hair_classes + eye_classes)
prev_state = torch.load(G_path)
G.load_state_dict(prev_state['model'])
G = G.eval()

def generate_by_attributes():
    hair_tag = torch.zeros(64, hair_classes).to(device)
    eye_tag = torch.zeros(64, eye_classes).to(device)
    hair_class = 7#np.random.randint(hair_classes)
    eye_class = 9#np.random.randint(eye_classes)
    
    for i in range(64):
        hair_tag[i][hair_class], eye_tag[i][eye_class] = 1, 1
    
    tag = torch.cat((hair_tag, eye_tag), 1)
    z = torch.randn(64, latent_dim).to(device)
    
    output = G(z, tag)
    save_image(utils.denorm(output), '../samples/{} hair {} eyes.png'.format(hair_mapping[hair_class], eye_mapping[eye_class]))

def hair_grad(n):
    eye = torch.zeros(eye_classes).to(device)
    eye[np.random.randint(eye_classes)] = 1
    eye.unsqueeze_(0)
    eye = torch.cat([eye for _ in range(batch_size)], dim = 0).to(device)
    
    z = torch.randn(latent_dim).unsqueeze(0).to(device)
    img_list = []
    for i in range(hair_classes):
        hair = torch.zeros(hair_classes).to(device)
        hair[i] = 1
        hair.unsqueeze_(0)
        tag = torch.cat((hair, eye), 1)
        img_list.append(G(z, tag))
        
    output = torch.cat(img_list, 0)
    print(output.shape)
    save_image(utils.denorm(output), '../samples/change_hair_color_{}.png'.format(n), nrow = hair_classes)

def eye_grad(n):
    hair = torch.zeros(hair_classes).to(device)
    hair[np.random.randint(hair_classes)] = 1
    hair.unsqueeze_(0)
    hair = torch.cat([hair for _ in range(batch_size)], dim = 0).to(device)
    
    z = torch.randn(latent_dim).unsqueeze(0).to(device)
    img_list = []
    for i in range(eye_classes):
        eye = torch.zeros(eye_classes).to(device)
        eye[i] = 1
        eye.unsqueeze_(0)
        tag = torch.cat((hair, eye), 1)
        img_list.append(G(z, tag))
        
    output = torch.cat(img_list, 0)
    print(output.shape)
    save_image(utils.denorm(output), '../samples/change_eye_color_{}.png'.format(n), nrow = eye_classes)

def fix_noise():
    z = torch.randn(latent_dim).unsqueeze(0).to(device)
    img_list = []
    for i in range(eye_classes):
        for j in range(hair_classes):
            eye = torch.zeros(eye_classes).to(device)
            hair = torch.zeros(hair_classes).to(device)
            eye[i], hair[j] = 1, 1
            eye.unsqueeze_(0)
            hair.unsqueeze_(0)
    
            tag = torch.cat((hair, eye), 1)
            img_list.append(G(z, tag))
        
    output = torch.cat(img_list, 0)
    save_image(utils.denorm(output), '../samples/fix_noise.png', nrow = hair_classes)
    
generate_by_attributes()
