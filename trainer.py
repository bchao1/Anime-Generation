import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
#from torch.utils.tensorboard import SummaryWriter

from models.ACGAN import Generator, Discriminator
from dataset.anime_dataset import get_anime_dataloader
from utils.utils import denorm, save_model

class ACGANTrainer:
    def __init__(self, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using {}".format(self.device))
        
        self.config = config
        self.run_dir = os.path.join('runs', str(config['run']))
        self.data_root = config['data_root']
        
        self.lr = config['optim']['lr']
        self.beta = config['optim']['beta']
        
        
        self.classes = config['classes']
        self.class_num = tuple(eval(config['class_num']))
        
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.print_n_iter = config['print_n_iter']
        self.sample_n_iter = config['sample_n_iter'] # sample generated image to save to file
        self.log_n_iter = config['log_n_iter']
        self.save_n_epoch = config['save_n_epoch']
        
        self.dataloader = get_anime_dataloader(self.data_root, self.class_num, self.batch_size)
        self.steps_per_epoch = int(np.ceil(self.dataloader.dataset.__len__() * 1.0 / self.batch_size))
        print("Traning images: {}".format(self.dataloader.dataset.__len__()))
            
        self.input_size = config['model']['input_size']
        self.noise_dim = config['model']['noise_dim']
        self.class_dim = config['model']['class_dim']
        
        self.G = Generator(self.noise_dim, self.class_dim).to(self.device)
        self.D = Discriminator(self.class_dim).to(self.device)
        
        self.dis_crit = nn.BCELoss() # discriminator criterion
        self.cls_crit = nn.BCELoss() # classifier criterion
        
        self.loss_weight = config['loss_weight']
        
        self.G_optim = optim.Adam(self.G.parameters(), lr = self.lr, betas = [self.beta, 0.999])
        self.D_optim = optim.Adam(self.D.parameters(), lr = self.lr, betas = [self.beta, 0.999])
        
        #self.writer = SummaryWriter(self.run_dir)
        
    def sample_class_label(self, batch_size):
        labels = []
        
        for c in self.class_num:
            label = torch.LongTensor(batch_size, 1).random_() % c
            one_hot = torch.zeros(batch_size, c).scatter(1, label, 1)
            labels.append(one_hot)
            
        labels = torch.cat(labels, 1)
        return labels
        
    
    def start(self):
        img, label, mask = next(iter(self.dataloader))
        print(img.shape)
        print(label.shape)
        print(mask.shape)
        
        l = self.sample_class_label(self.batch_size)
        print(l.shape)
        print(self.steps_per_epoch)
        
        itr = 0
        fix_z = torch.randn(self.batch_size, self.noise_dim).to(self.device)
        fix_class = self.sample_class_label(self.batch_size).to(self.device)
        
        for e in range(self.epochs):
            
            for i, (real_img, real_class, mask) in enumerate(self.dataloader):
                
                self.G.train()
                
                itr += 1
                
                fake_label = torch.zeros(self.batch_size).to(self.device)
                
                real_img = real_img.to(self.device)
                real_class = real_class.to(self.device)
                mask = mask.to(self.device)
                
                # Train D
                real_label = torch.empty(self.batch_size).uniform_(0.9, 1).to(self.device) # one - sided smoothing
                fake_class = self.sample_class_label(self.batch_size).to(self.device)
                z = torch.randn(self.batch_size, self.noise_dim).to(self.device)

                fake_img = self.G(z, fake_class).to(self.device)
                
                real_score, real_pred = self.D(real_img)
                fake_score, fake_pred = self.D(fake_img)
                
                real_dis_loss = self.dis_crit(real_score, real_label)
                fake_dis_loss = self.dis_crit(fake_score, fake_label)
                dis_loss = (real_dis_loss + fake_dis_loss) * 0.5
                
                real_cls_loss = self.cls_crit(real_pred * mask, real_class * mask)
                
                D_loss = real_cls_loss * self.loss_weight['cls'] + dis_loss * self.loss_weight['dis']
                
                self.D_optim.zero_grad()
                D_loss.backward()
                self.D_optim.step()
                
                # Train G
                real_label = torch.ones(self.batch_size).to(self.device)
                fake_class = self.sample_class_label(self.batch_size).to(self.device)
                z = torch.randn(self.batch_size, self.noise_dim).to(self.device)

                fake_img = self.G(z, fake_class).to(self.device)
                fake_score, fake_pred = self.D(fake_img)
                
                fake_dis_loss = self.dis_crit(fake_score, real_label)
                fake_cls_loss = self.cls_crit(fake_pred, fake_class)
                G_loss = fake_cls_loss * self.loss_weight['cls'] + fake_dis_loss * self.loss_weight['dis']
                
                
                cls_loss = (fake_cls_loss + real_cls_loss) * 0.5
                
                self.G_optim.zero_grad()
                G_loss.backward()
                self.G_optim.step()
                
                if itr % self.print_n_iter == 0:
                    print("| Epoch {} | {} / {} | D: {} | G: {} | cls: {} |".format(e + 1, i + 1, self.steps_per_epoch, D_loss.item(), G_loss.item(), cls_loss.item()))
                
                """
                if itr % self.log_n_iter == 0:
                    self.writer.add_scalar('loss/D', D_loss.item(), itr)
                    self.writer.add_scalar('loss/G', G_loss.item(), itr)
                    self.writer.add_scalar('loss/cls', cls_loss.item(), itr)
                """
                
                if itr % self.sample_n_iter == 0:
                    self.G.eval()
                    
                    fixed_img = denorm(self.G(fix_z, fix_class))
                    
                    z = torch.randn(self.batch_size, self.noise_dim).to(self.device)
                    c = self.sample_class_label(1).repeat(self.batch_size, 1).to(self.device)
                    class_img = denorm(self.G(z, c))
                    
                    save_image(fixed_img, os.path.join(self.run_dir, 'images', 'fix', '{}.png'.format(itr)))
                    save_image(class_img, os.path.join(self.run_dir, 'images', 'class', '{}.png'.format(itr)))
                    #self.writer.add_image('fix', fixed_img, itr)
                    #self.writer.add_image('class', class_img, itr)
                    
            if (e + 1) % self.save_n_epoch == 0:
                save_model(self.G, self.G_optim, os.path.join(self.run_dir, 'ckpt', 'G_{}.pth'.format(itr)))
                save_model(self.D, self.D_optim, os.path.join(self.run_dir, 'ckpt', 'D_{}.pth'.format(itr)))
                
                