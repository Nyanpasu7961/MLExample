'''
Processes images from 2 folders and saves kuwahara image only if they are structurally similar i.e. SSIM > 0.9.
'''
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread, imsave
from skimage.color import rgb2gray
import os
import numpy as np
from pathlib import Path

THRESHOLD = 0.9

data_types = ['trainA', 'trainB']

training_data = 'datasets/TRAIN_DATA/'
kuwahara_data = 'datasets/kuwahara_data/'

output_folder = 'datasets/actual_finetune'
os.makedirs(output_folder, exist_ok=True)

def compute_ssim(img1_path, img2_path):
    img1 = rgb2gray(imread(img1_path))
    img2 = rgb2gray(imread(img2_path))
    score, _ = ssim(img1, img2, full=True, data_range=1.0)
    return score


for d in data_types:
    t_path = training_data+d
    k_path = kuwahara_data+d

    for filename in os.listdir(t_path):
        path1 = os.path.join(t_path, filename)
        path2 = os.path.join(k_path, filename)

        if os.path.exists(path2):
            try:
                score = compute_ssim(path1, path2)
                if score > THRESHOLD:
                    imsave(os.path.join(output_folder, filename), imread(path2))
                    print(f"Saved {filename+d} with SSIM = {score:.4f}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Missing: {filename} in {kuwahara_data}")