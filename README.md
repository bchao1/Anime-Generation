# Anime-Generation
A very simple DCGAN-based baseline model for anime generation using ACGAN.
## Usage
### Training 
The directory tree is organized as follows:
```
├── Anime Generation
    ├── src
    │    ├── train.py
    │    ├── datasets.py
    │    ├── ACGAN.py
    │    ├── utils.py
    │
    ├── {training data}
    │    ├── images  # Training images
    │    ├── tags.pickle  # One-hot encoding tags for training images
    │    ├── hair_prob.npy  # Probability distribution for hair classes.
    │    ├── eye_prob.npy  # Probability distribution for eye classes.
    │
    ├── {generated samples}
    │    ├──{train event}
    │        ├── real.png
    │        ├── fake_step_100.png
    │        ├── ...
    │        
    ├──checkpoints
        ├──{train event}
            ├── loss.png
            ├── classifier_loss.png
            ├── G_10000.ckpt
            ├── D_10000.ckpt
            ├── ....
```

Please pass your own training data folder, generated samples folder, and checkpoint folder as arguments when running `train.py`. Default folders are `data`, `samples`, and `checkpoints`.  
  
Images are tagged with the following labels:  
- Hair tags (12 classes)
```python
['orange', 'white', 'aqua', 'gray', 'green', 'red', 'purple', 'pink', 'blue', 'black', 'brown', 'blonde']
```
- Eye tags (10 classes)
```python
['black', 'orange', 'pink', 'yellow', 'aqua', 'purple', 'green', 'brown', 'red', 'blue']
```
Each training image is tagged with exactly two labels: one from hair tags and the other from eye tags. In `tags.pickle`, each image index is associated with a 22-dimensional tensor. 
```python
{1: tensor([1.0, 0, 0, 0, ......., 1.0, 0, 0]),
 2: tensor([0, 0, 1.0, 0, ......., , 0, 0, 1.0],
 ......}
```
You can find the training data here: https://drive.google.com/file/d/1jdJXkQIWVGOeb0XJIXE3YuZQeiEPd8rM/view?usp=sharing .
***
### Testing
## Model Architecture
### DCGAN 

![DCGAN_Structure](./img_src/DCGAN.png)
For more details on DCGAN, please refer to https://github.com/Mckinsey666/GAN-Tutorial and the original paper: https://arxiv.org/abs/1511.06434.
### ACGAN 

![ACGAN_Structure](./img_src/ACGAN.png)  
In ACGAN, the discriminator not only learns to discriminate between real and synthesized images, but also needs to classify images into different classes. This not only stablizes training, but also allows us to manipulate generated image attribures.  

For more information, please refer to the original ACGAN paper: https://arxiv.org/abs/1610.09585.
## Algorithm
![ACGAN algorithm](./img_src/algo.png)
Some modifications are made:
1. The classification loss for the synthesized image was omitted, since we believe the fake image would confuse the discriminator.
2. The adversarial loss for the discriminator was divided by 2 (average of the fake adversarial loss and real adversarial loss)
## Results
Fixed noise, change eye and hair colors.

![fixed noise](./results/fix_noise.png)
***
Fixed eye attribute and noise, change hair colors.

![change hair color](./results/change_hair_color.png)
***
Fixed hair attribute and noise, change eye colors.

![change eye color](./results/change_eye_color.png)
***
Fixed hair and eye attributes, change noise.

- Orange hair green eyes

![orange hair green eyes](./results/orange_hair_green_eyes.png)

- Pink hair blue eyes

![pink hair blue eyes](./results/pink_hair_blue_eyes.png)

- White hair purple eyes

![white hair purple eyes](./results/white_hair_purple_eyes.png)

## Improvements to be made
- Low color intensity
    - Try removing batch norm in discriminator
    - Might be a type of mode collapse


