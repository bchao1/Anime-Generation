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
import ACGAN
import utils
from argparse import ArgumentParser

hair_mapping =  ['orange', 'white', 'aqua', 'gray', 'green', 'red', 'purple', 
                 'pink', 'blue', 'black', 'brown', 'blonde']
hair_dict = {
    'orange' : 0,
    'white': 1, 
    'aqua': 2,
    'gray': 3,
    'green': 4,
    'red': 5,
    'purple': 6,
    'pink': 7,
    'blue': 8,
    'black': 9,
    'brown': 10,
    'blonde': 11
}

eye_mapping = ['black', 'orange', 'pink', 'yellow', 'aqua', 'purple', 'green', 
               'brown', 'red', 'blue']
eye_dict = {
    'black': 0,
    'orange': 1,
    'pink': 2,
    'yellow': 3,
    'aqua' : 4,
    'purple': 5,
    'green': 6,
    'brown': 7,
    'red': 8,
    'blue': 9
}


parser = ArgumentParser()
parser.add_argument('-t', '--type', help = 'Type of anime generation.', 
                    choices = ['fix_noise', 'fix_hair_eye', 'change_hair', 'change_eye', 'interpolate'], 
                    default = 'fix_noise', type = str)
parser.add_argument('--hair', help = 'Determine the hair color of the anime characters.', 
                    default = None, choices = hair_mapping, type = str)
parser.add_argument('--eye',  help = 'Determine the eye color of the anime characters.',
                    default = None, choices = eye_mapping, type = str)
parser.add_argument('-s', '--sample_dir', help = 'Folder to save the generated samples.',
                    default = '../generated', type = str)
parser.add_argument('-d', '--model_dir', help = 'Folder where the trained model is saved',
                    default = '../models', type = str)
args = parser.parse_args()

def generate_by_attributes(model, device, latent_dim, hair_classes, eye_classes, hair_color, eye_color):
    hair_tag = torch.zeros(64, hair_classes).to(device)
    eye_tag = torch.zeros(64, eye_classes).to(device)
    hair_class = hair_dict[hair_color]
    eye_class = eye_dict[eye_color]
    for i in range(64):
        hair_tag[i][hair_class], eye_tag[i][eye_class] = 1, 1
    
    tag = torch.cat((hair_tag, eye_tag), 1)
    z = torch.randn(64, latent_dim).to(device)
    
    output = model(z, tag)
    save_image(utils.denorm(output), '{}/{} hair {} eyes.png'.format(args.sample_dir, hair_mapping[hair_class], eye_mapping[eye_class]))

def hair_grad(model, device, latent_dim, hair_classes, eye_classes):
    eye = torch.zeros(eye_classes).to(device)
    eye[np.random.randint(eye_classes)] = 1
    eye.unsqueeze_(0)
    
    z = torch.randn(latent_dim).unsqueeze(0).to(device)
    img_list = []
    for i in range(hair_classes):
        hair = torch.zeros(hair_classes).to(device)
        hair[i] = 1
        hair.unsqueeze_(0)
        tag = torch.cat((hair, eye), 1)
        img_list.append(model(z, tag))
        
    output = torch.cat(img_list, 0)
    save_image(utils.denorm(output), '{}/change_hair_color.png'.format(args.sample_dir), nrow = hair_classes)

def eye_grad(model, device, latent_dim, hair_classes, eye_classes):
    hair = torch.zeros(hair_classes).to(device)
    hair[np.random.randint(hair_classes)] = 1
    hair.unsqueeze_(0)
    
    z = torch.randn(latent_dim).unsqueeze(0).to(device)
    img_list = []
    for i in range(eye_classes):
        eye = torch.zeros(eye_classes).to(device)
        eye[i] = 1
        eye.unsqueeze_(0)
        tag = torch.cat((hair, eye), 1)
        img_list.append(model(z, tag))
        
    output = torch.cat(img_list, 0)
    save_image(utils.denorm(output), '{}/change_eye_color.png'.format(args.sample_dir), nrow = eye_classes)

def fix_noise(model, device, latent_dim, hair_classes, eye_classes):
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
            img_list.append(model(z, tag))
        
    output = torch.cat(img_list, 0)
    save_image(utils.denorm(output), '{}/fix_noise.png'.format(args.sample_dir), nrow = hair_classes)

def interpolate(model, device, latent_dim, hair_classes, eye_classes, samples = 10):
    z1 = torch.randn(1, latent_dim).to(device)
    h1 = torch.zeros(1, hair_classes).to(device)
    e1 = torch.zeros(1, eye_classes).to(device)
    h1[0][np.random.randint(hair_classes)] = 1
    e1[0][np.random.randint(eye_classes)] = 1    
    c1 = torch.cat((h1, e1), 1)
    
    z2 = torch.randn(1, latent_dim).to(device)
    h2 = torch.zeros(1, hair_classes).to(device)
    e2 = torch.zeros(1, eye_classes).to(device)
    h2[0][np.random.randint(hair_classes)] = 1
    e2[0][np.random.randint(eye_classes)] = 1    
    c2 = torch.cat((h2, e2), 1)
    
    z_diff = z2 - z1
    c_diff = c2 - c1
    z_step = z_diff / (samples + 1)
    c_step = c_diff / (samples + 1)
    
    img_list = []
    for i in range(0, samples + 2):
        z = z1 + z_step * i
        c = c1 + c_step * i
        img_list.append(model(z, c))
    output = torch.cat(img_list, 0)
    save_image(utils.denorm(output), '{}/interpolation.png'.format(args.sample_dir), nrow = samples + 2)
        
def main():
    if not os.path.exists(args.sample_dir):
        os.mkdir(args.sample_dir)
    latent_dim = 100
    hair_classes = 12
    eye_classes = 10
    batch_size = 1

    device = 'cpu'
    G_path = '{}/ACGAN_generator.ckpt'.format(args.model_dir)

    G = G = ACGAN.Generator(latent_dim = latent_dim, class_dim = hair_classes + eye_classes)
    prev_state = torch.load(G_path)
    G.load_state_dict(prev_state['model'])
    G = G.eval()

    if args.type == 'fix_hair_eye':
        generate_by_attributes(G, device, latent_dim, hair_classes, eye_classes, args.hair,  args.eye)
    elif args.type == 'change_eye':
        eye_grad(G, device, latent_dim, hair_classes, eye_classes)
    elif args.type == 'change_hair':
        hair_grad(G, device, latent_dim, hair_classes, eye_classes)
    elif args.type == 'interpolate':
        interpolate(G, device, latent_dim, hair_classes, eye_classes)
    else:
        fix_noise(G, device, latent_dim, hair_classes, eye_classes)
    
if __name__ == "__main__":
    main()