import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

from models.ACGAN import Generator, Discriminator
from dataset.anime_dataset import get_anime_h5_dataloader
from utils.utils import denorm, save_model, Logger

class ACGANTrainer_h5:
    def __init__(self, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using {}".format(self.device))
        if self.device == "cuda":
            torch.cuda.set_device(config["gpu_id"])
        
        self.config = config
        self.run_dir = config["run_dir"]
        self.save_dir = config["save_dir"]
        self.data_root = config['data_root']
        
        self.lr = config['optim']['lr']
        self.beta = config['optim']['beta']
        
        self.class_num = int(config['class_num'][0])
        self.select_classes = config["select_classes"]
        if self.select_classes is not None:
            assert len(self.select_classes) == self.class_num
        
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.print_n_iter = config['print_n_iter']
        self.sample_n_iter = config['sample_n_iter'] # sample generated image to save to file
        self.log_n_iter = config['log_n_iter']
        self.save_n_epoch = config['save_n_epoch']
        
        self.dataloader = get_anime_h5_dataloader(self.data_root, self.batch_size, self.select_classes)
        self.steps_per_epoch = int(np.ceil(self.dataloader.dataset.__len__() * 1.0 / self.batch_size))
        print("Traning images: {}".format(self.dataloader.dataset.__len__()))
            
        self.input_size = config['model']['input_size']
        self.noise_dim = config['model']['noise_dim']
        self.class_dim = self.class_num
        
        self.G = Generator(self.noise_dim, self.class_dim).to(self.device)
        self.D = Discriminator(self.class_dim, "softmax").to(self.device)
        
        self.dis_crit = nn.BCELoss() # discriminator criterion
        self.cls_crit = nn.BCELoss() # classifier criterion
        
        self.loss_weight = config['loss_weight']
        
        self.G_optim = optim.Adam(self.G.parameters(), lr = self.lr, betas = [self.beta, 0.999])
        self.D_optim = optim.Adam(self.D.parameters(), lr = self.lr, betas = [self.beta, 0.999])
        
        self.writer = SummaryWriter(self.run_dir)
        sys.stdout = Logger()
        
    def sample_class_label(self, batch_size):
        label = torch.LongTensor(batch_size, 1).random_() % self.class_num
        one_hot = torch.zeros(batch_size, self.class_num).scatter(1, label, 1)
        return one_hot
    
        
    
    def start(self):
        img, label = next(iter(self.dataloader))
        l = self.sample_class_label(self.batch_size)
       
        print(img.shape, label.shape, l.shape)
        
        itr = 0
        samples = 8
        fix_z = torch.repeat_interleave(torch.randn(samples, self.noise_dim), self.class_num, 0).to(self.device)
        #fix_class = self.sample_class_label(self.batch_size).to(self.device)
        fix_class = torch.eye(self.class_num).repeat(samples, 1).to(self.device)

        
        
        for e in range(self.epochs): 
            for i, (real_img, real_class) in enumerate(self.dataloader):
                
                self.G.train()
                
                itr += 1
                
                fake_label = torch.zeros(self.batch_size).to(self.device).float()
                
                real_img = real_img.to(self.device).float()
                real_class = real_class.to(self.device).float()
                
                # Train D
                real_label = torch.empty(self.batch_size).uniform_(0.9, 1).to(self.device) # one - sided smoothing
                fake_class = self.sample_class_label(self.batch_size).to(self.device)
                z = torch.randn(self.batch_size, self.noise_dim).to(self.device)

                fake_img = self.G(z, fake_class).to(self.device)
                
                real_score, real_pred = self.D(real_img)
                fake_score, fake_pred = self.D(fake_img)
                
                real_dis_loss = self.dis_crit(real_score, real_label)
                fake_dis_loss = self.dis_crit(fake_score, fake_label)
                dis_loss = (real_dis_loss + fake_dis_loss)
                
                real_cls_loss = self.cls_crit(real_pred, real_class)
                fake_cls_loss = self.cls_crit(fake_pred, fake_class)
                cls_loss = (real_cls_loss + fake_cls_loss)
                
                D_loss = cls_loss * self.loss_weight['cls'] + dis_loss * self.loss_weight['dis']
                
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
                
                
                self.G_optim.zero_grad()
                G_loss.backward()
                self.G_optim.step()
                
                if itr % self.print_n_iter == 0:
                    print("| Epoch {} | {} / {} | D: {} | G: {} | cls: {} |".format(e + 1, i + 1, self.steps_per_epoch, D_loss.item(), G_loss.item(), cls_loss.item()))
                
                self.writer.add_scalar('loss/D', D_loss.item(), itr)
                self.writer.add_scalar('loss/G', G_loss.item(), itr)
                self.writer.add_scalar('loss/cls', cls_loss.item(), itr)
                
                if itr % self.sample_n_iter == 0:
                    self.G.eval()

                    class_img = denorm(self.G(fix_z, fix_class))
                    
                    save_image(class_img, os.path.join(self.save_dir, 'images', 'class', '{}.png'.format(itr)), nrow=self.class_num, padding=2, pad_value=255)
                    
                    image_grid = make_grid(class_img, nrow=self.class_num, padding=2, pad_value=255)
                    self.writer.add_image('fix', image_grid, itr)
                    #self.writer.add_image('class', class_img, itr)
            

            if (e + 1) % self.save_n_epoch == 0:
                save_model(self.G, self.G_optim, os.path.join(self.save_dir, 'ckpt', 'G_{}.pth'.format(itr)))
                save_model(self.D, self.D_optim, os.path.join(self.save_dir, 'ckpt', 'D_{}.pth'.format(itr)))
                
                