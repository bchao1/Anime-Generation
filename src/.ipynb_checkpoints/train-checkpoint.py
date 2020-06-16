import torch
import torch.nn
import torch.optim as optim
import torchvision.transforms as Transform
from torchvision.utils import save_image

import numpy as np
import os
from argparse import ArgumentParser

from datasets import Anime, Shuffler
from ACGAN import Generator, Discriminator
from utils import save_model, denorm, plot_loss, plot_classifier_loss, show_process
from utils import generation_by_attributes, get_random_label

parser = ArgumentParser()
parser.add_argument('-d', '--device', help = 'Device to train the model on', 
                    default = 'cpu', choices = ['cuda', 'cpu'], type = str)
parser.add_argument('-i', '--iterations', help = 'Number of iterations to train ACGAN', 
                    default = 50000, type = int)
parser.add_argument('-b', '--batch_size', help = 'Training batch size',
                    default = 128, type = int)
parser.add_argument('-t', '--train_dir', help = 'Training data directory', 
                    default = '../data', type = str)
parser.add_argument('-s', '--sample_dir', help = 'Directory to store generated images', 
                    default = '../samples', type = str)
parser.add_argument('-c', '--checkpoint_dir', help = 'Directory to save model checkpoints', 
                    default = '../checkpoints', type = str)
parser.add_argument('--sample', help = 'Sample every _ steps', 
                    default = 500, type = int)
parser.add_argument('--check', help = 'Save model every _ steps', 
                    default = 2000, type = int)
parser.add_argument('--lr', help = 'Learning rate of ACGAN. Default: 0.0002', 
                    default = 0.0002, type = float)
parser.add_argument('--beta', help = 'Momentum term in Adam optimizer. Default: 0.5', 
                    default = 0.5, type = float)
parser.add_argument('--aux', '--classification_weight', help = 'Classification loss weight. Default: 1',
                    default = 1, type = float)
args = parser.parse_args()

if args.device == 'cuda' and not torch.cuda.is_available():
    print("Your device currenly doesn't support CUDA.")
    exit()
print('Using device: {}'.format(args.device))

def main():
    batch_size = args.batch_size
    iterations =  args.iterations
    device = args.device
    
    hair_classes, eye_classes = 12, 10
    num_classes = hair_classes + eye_classes
    latent_dim = 100
    
    config = 'ACGAN-batch_size-[{}]-steps-[{}]'.format(batch_size, iterations)
    print('Configuration: {}'.format(config))
    
    
    root_dir = '../{}/images'.format(args.train_dir)
    tags_file = '../{}/tags.pickle'.format(args.train_dir)
    hair_prior = np.load('../{}/hair_prob.npy'.format(args.train_dir))
    eye_prior = np.load('../{}/eye_prob.npy'.format(args.train_dir))

    random_sample_dir = '../{}/{}/random_generation'.format(args.sample_dir, config)
    fixed_attribute_dir = '../{}/{}/fixed_attributes'.format(args.sample_dir, config)
    checkpoint_dir = '../{}/{}'.format(args.checkpoint_dir, config)
    
    if not os.path.exists(random_sample_dir):
    	os.makedirs(random_sample_dir)
    if not os.path.exists(fixed_attribute_dir):
    	os.makedirs(fixed_attribute_dir)
    if not os.path.exists(checkpoint_dir):
    	os.makedirs(checkpoint_dir)
        
    ########## Start Training ##########

    transform = Transform.Compose([Transform.ToTensor(),
                                   Transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = Anime(root_dir = root_dir, tags_file = tags_file, transform = transform)
    shuffler = Shuffler(dataset = dataset, batch_size = args.batch_size)
    
    G = Generator(latent_dim = latent_dim, class_dim = num_classes).to(device)
    D = Discriminator(num_classes = num_classes).to(device)

    G_optim = optim.Adam(G.parameters(), betas = [args.beta, 0.999], lr = args.lr)
    D_optim = optim.Adam(D.parameters(), betas = [args.beta, 0.999], lr = args.lr)
    
    d_log, g_log, classifier_log = [], [], []
    criterion = torch.nn.BCELoss()

    for step_i in range(1, iterations + 1):

        real_label = torch.ones(batch_size).to(device)
        fake_label = torch.zeros(batch_size).to(device)
        
        # Train discriminator
        real_img, hair_tags, eye_tags = shuffler.get_batch()
        real_img, hair_tags, eye_tags = real_img.to(device), hair_tags.to(device), eye_tags.to(device)
        real_tag = torch.cat((hair_tags, eye_tags), dim = 1)

        z = torch.randn(batch_size, latent_dim).to(device)
        fake_tag = get_random_label(batch_size = batch_size, 
                                    hair_classes = hair_classes, hair_prior = hair_prior,
                                    eye_classes = eye_classes, eye_prior = eye_prior).to(device)
        fake_img = G(z, fake_tag).to(device)
                
        real_score, real_predict = D(real_img)
        fake_score, fake_predict = D(fake_img)
            
        real_discrim_loss = criterion(real_score, real_label)
        fake_discrim_loss = criterion(fake_score, fake_label)

        real_classifier_loss = criterion(real_predict, real_tag)
        
        discrim_loss = (real_discrim_loss + fake_discrim_loss) * 0.5
        classifier_loss = real_classifier_loss * args.classification_weight
        
        classifier_log.append(classifier_loss.item())
            
        D_loss = discrim_loss + classifier_loss
        D_optim.zero_grad()
        D_loss.backward()
        D_optim.step()

        # Train generator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_tag = get_random_label(batch_size = batch_size, 
                                    hair_classes = hair_classes, hair_prior = hair_prior,
                                    eye_classes = eye_classes, eye_prior = eye_prior).to(device)
        fake_img = G(z, fake_tag).to(device)
        
        fake_score, fake_predict = D(fake_img)
        
        discrim_loss = criterion(fake_score, real_label)
        classifier_loss = criterion(fake_predict, fake_tag)
        
        G_loss = classifier_loss + discrim_loss
        G_optim.zero_grad()
        G_loss.backward()
        G_optim.step()
            
        ########## Updating logs ##########
        d_log.append(D_loss.item())
        g_log.append(G_loss.item())
        show_process(total_steps = iterations, step_i = step_i,
        			 g_log = g_log, d_log = d_log, classifier_log = classifier_log)

        ########## Checkpointing ##########

        if step_i == 1:
            save_image(denorm(real_img[:64,:,:,:]), os.path.join(random_sample_dir, 'real.png'))
        if step_i % args.sample == 0:
            save_image(denorm(fake_img[:64,:,:,:]), os.path.join(random_sample_dir, 'fake_step_{}.png'.format(step_i)))
            
        if step_i % args.check == 0:
            save_model(model = G, optimizer = G_optim, step = step_i, log = tuple(g_log), 
                       file_path = os.path.join(checkpoint_dir, 'G_{}.ckpt'.format(step_i)))
            save_model(model = D, optimizer = D_optim, step = step_i, log = tuple(d_log), 
                       file_path = os.path.join(checkpoint_dir, 'D_{}.ckpt'.format(step_i)))

            plot_loss(g_log = g_log, d_log = d_log, file_path = os.path.join(checkpoint_dir, 'loss.png'))
            plot_classifier_loss(log = classifier_log, file_path = os.path.join(checkpoint_dir, 'classifier loss.png'))

            generation_by_attributes(model = G, device = args.device, step = step_i, latent_dim = latent_dim, 
                                     hair_classes = hair_classes, eye_classes = eye_classes, 
                                     sample_dir = fixed_attribute_dir)
    
if __name__ == '__main__':
    main()
            

        

    
    
    