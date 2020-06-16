# ACGAN Conditional Anime Generation
> Note: I have restructured the whole folder, cleaned up training code, pruned my dataset, and updated most of the results. You can see the old version of this repo in `old`.

Generate colorful anime characters using GAN.
<br>
<p align="center">
  <img src="results/change_hair.png" />
</p>
<p align="center">
  <img src="results/change_eye.png" />
</p>

By interpolating generator input, we can see some interesting results.
<p align="center">
  <img src="results/interpolate.png" />
</p>

By fixing color classes and varying noise, we can generate anime characters with same colors but different identity.

<p align="center">
  <img src="results/blue_eye_red_hair.png" />
</p>
<p align="center">
  <img src="results/green_eye_purple_hair.png" />
</p>

<p align="center">
  <img src="results/orange_eye_blue_hair.png" />
</p>

<p align="center">
  <img src="results/purple_eye_pink_hair.png" />
</p>

<p align="center">
  <img src="results/red_eye_black_hair.png" />
</p>

## Start traning
Modify `config.yaml` as you wish.
```
> python3 run.py
```
## The dataset
The current [dataset](https://drive.google.com/file/d/18FG-KSeAUBp0O0np63ynqdA7_tgIOCa8/view?usp=sharing) is a composition of 2 datasets:

1. One dataset with eye and hair color labels (30k+ images)
2. One dataset with year label, which is the year in which the anime is produced (60k+ images).

The dataset format is as follows:
```
- images/
    - XXXXX.jpg
    - ...
- labels.pkl
- eye_label.json
- year_label.json
- hair_label.json
```
After loading in `labels.pkl` with pickle, you will get a dictionary of `{ filname : labels }`. The labels are formatted as `(eye, hair, year)` tuples.
```
{
    "32455": (8, 10, 5),
    ...
}
```
> This means `32455.jpg` has eye class 8, hair class 10, year class 5.

Missing labels will be a `None`. All images from dataset 1 will have year labels `None`, while all images from dataset 2 will have eye and hair label `None`.
***
Source code in the current repo is used to train on the first dataset. This requires some manual preprocessing (see `dataset/anime_dataset.py`) to extract the first dataset from the whole dataset. 
***
The `.json` files map discrete labels to semantics. 
```js
// eye_label.json
{
    "aqua": 0, 
    "black": 1, 
    ...
}
```
## Some notes
- When training on the first dataset, adding some color transformations to images as a preprocessing step traning might help. You can achieve this through various `torchvisions.transforms.functional` methods.
- Train with N(0, 1) but sample from Gaussian of smaller variance when evaluating. Just an engineering hack to get better results.