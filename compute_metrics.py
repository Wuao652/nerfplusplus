# This file is used to evaluate the haze removal performance.
# Computing the metrics including PSNR, SSIM.
import os
import glob
import imageio
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def find_files(dir, exts):
    if os.path.isdir(dir):
        # types should be ['*.png', '*.jpg']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []

gt_path = '/home/dennis/nerfplusplus/data/carla_data/gt/9actors'
gt_img_files = find_files(
    gt_path,
    exts=['*.png', '*.jpg'])

dehazed_img_path = '/home/dennis/image_dehaze/results/9actors/radiance'
dehazed_img_files = find_files(
    dehazed_img_path,
    exts=['*.png', '*.jpg'])

img_cnt = len(dehazed_img_files)
psnr_all = np.zeros(img_cnt)
ssim_all = np.zeros(img_cnt)
for dehazed_img_file in dehazed_img_files:
    dehazed_img_base = os.path.basename(dehazed_img_file)
    for gt_img_file in gt_img_files:
        gt_img_base = os.path.basename(gt_img_file)
        if dehazed_img_base[-11:] == gt_img_base:
            dehazed_img = imageio.imread(dehazed_img_file).astype(np.float32) / 255.
            gt_img = imageio.imread(gt_img_file).astype(np.float32) / 255.
            assert (dehazed_img.shape == gt_img.shape)
            psnr = peak_signal_noise_ratio(gt_img, dehazed_img)
            ssim = structural_similarity(gt_img, dehazed_img, multichannel=True)
            idx = int(dehazed_img_base[-9:-4])
            psnr_all[idx] = psnr
            ssim_all[idx] = ssim
            print('psnr: ', psnr)
            print('ssim: ', ssim)
print('The overall PSNR is ', psnr_all.mean())
print('The overall SSIM is ', ssim_all.mean())
