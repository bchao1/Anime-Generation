import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from models.ACGAN import Generator, Discriminator
from utils.utils import load_model, denorm

torch.manual_seed(0)

noise_dim = 100
class_dim = 2

device = "cuda"
torch.cuda.set_device(0)

G = Generator(noise_dim, class_dim).to(device)
D = Discriminator(class_dim, "softmax").to(device)

G_path = "/mnt/data3/bchao/anime/experiments/acgan_h5_v8/ckpt/G_57450.pth"
D_path = "/mnt/data3/bchao/anime/experiments/acgan_h5_v8/ckpt/D_57450.pth"

G = load_model(G, None, G_path)
D = load_model(D, None, D_path)

# for cherry-picking
total_syn_img = []
total_score = []

# total batches * samples_per_batch images
batches = 1
samples_per_batch = 100
for _ in tqdm(range(batches)):
    with torch.no_grad():
        fix_z = torch.repeat_interleave(torch.normal(0, 1, size=(samples_per_batch, noise_dim)), class_dim, 0).to(device)
        fix_class = torch.eye(class_dim).repeat(samples_per_batch, 1).to(device)
        syn_img = G(fix_z, fix_class)
        score, _ = D(syn_img)
    total_syn_img.append(syn_img)
    total_score.append(score)
total_syn_img = denorm(torch.cat(total_syn_img))
total_score = torch.cat(total_score)
total_images = samples_per_batch * batches
total_syn_img = total_syn_img.view(total_images, class_dim, *total_syn_img.shape[1:])
#for i in tqdm(range(total_images)):
#    save_image(total_syn_img[i], f"results/by_year/{i}.png", nrow=class_dim, padding=2, pad_value=255)

select_id = [10, 13, 14, 35, 60, 65, 72, 82, 86, 94, 98]
selected_imgs = total_syn_img[select_id] # (selected_images, class_dim, c, h, w)
print(selected_imgs.shape)
selected_imgs = torch.permute(selected_imgs, (1, 0, 2, 3, 4)).reshape(class_dim * len(select_id), *selected_imgs.shape[2:])
save_image(selected_imgs, f"results/year_selected.png", nrow=len(select_id), padding=2, pad_value=255)
exit()

avg_score = 0
for i in range(class_dim):
    avg_score += total_score[i::class_dim]
avg_score /= class_dim


topk = 10
top_scores, top_samples_id = torch.topk(avg_score, topk)
print(top_scores, top_samples_id)

results = []
for i in top_samples_id:
    results.append(total_syn_img[class_dim * i:class_dim * (i + 1)])
results = torch.cat(results, 0)

save_image(results, "results/by_year.png", nrow=class_dim, padding=2, pad_value=255)




