import os
import json
import pickle
from tqdm import tqdm
import shutil
import numpy as np
from PIL import Image
import h5py

root = "/mnt/data2/bchao/anime/data/"

def generate_anime_dataset(filepath, imsize=64):
    with open(os.path.join(root, "labels.pkl"), "rb") as file:
        labels_dict = pickle.load(file)

    year_imgs = []
    hair_eye_imgs = []
    year_labels = []
    hair_eye_labels = []
    for i, (img_id, class_labels) in enumerate(tqdm(labels_dict.items())):
        img_filename = img_id + ".jpg"
        img_filepath = os.path.join(root, "images", img_filename)
        if os.path.exists(img_filepath):
            img = np.array(Image.open(img_filepath).resize((imsize, imsize)))
            if class_labels[0] is not None and class_labels[1] is not None and class_labels[2] is None:
                # Save to hair and history group
                hair_eye_imgs.append(img)
                hair_eye_labels.append(list(class_labels[:2]))
            elif class_labels[0] is None and class_labels[1] is None and class_labels[2] is not None:
                # Save to year group
                year_imgs.append(img)
                year_labels.append(class_labels[2])
    year_imgs = np.stack(year_imgs)
    hair_eye_imgs = np.stack(hair_eye_imgs)
    year_labels = np.array(year_labels)
    hair_eye_labels = np.array(hair_eye_labels)

    with h5py.File(filepath, 'w') as hf:
        dset_year_imgs = hf.create_dataset("year_imgs", data=year_imgs)
        dset_hair_eye_imgs = hf.create_dataset("hair_eye_imgs", data=hair_eye_imgs)
        dset_year_labels = hf.create_dataset("year_labels", data=year_labels)
        dset_hair_eye_labels = hf.create_dataset("hair_eye_labels", data=hair_eye_labels)



def test_anime_dataset():
    pass

if __name__ == "__main__":
    generate_anime_dataset(os.path.join(root, "dataset.h5"), 64)