import numpy as np


def rgb2Class(mask_rgb):
  
    color2class = {
    #(R, G, B): int # color - label
    (0, 0, 0): 0, # black - ground
    (255, 0, 0): 1, # red - sky
    (0, 255, 0): 2, # green - small rock
    (0, 0, 255): 3 # blue - big rock
    }

    # make a new array that's the same height and width as the original photo
    # fill it with 255 so that we know if there's missing pixels that need to be imputed
    H, W, _ = mask_rgb.shape
    output = np.full((H,W), 255, np.uint8)

    # for each color in the dictionary find all the pixels that match that color and make a mask
    # for the masked pixels set it to that class
    for (r, g, b), clss in color2class.items():
        matches = (mask_rgb[...,0] == r) & (mask_rgb[...,1] == g) & (mask_rgb[...,2] == b)
        output[matches] = clss

    # return new mask
    return output

import os, cv2 as cv, torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Build an augmentation pipeline:
train_aug = A.Compose([
    # Downscale (resize to smaller resolution)
    A.Resize(height=128, width=192, interpolation=cv.INTER_AREA),
    # Spatial augs:
    A.HorizontalFlip(p=0.5),
    A.RandomResizedCrop(size=(64,96), scale=(0.5, 1.0), ratio=(0.9, 1.1), p=0.8),
    # Convert to tensors
    ToTensorV2(transpose_mask=True)
])

from torch.utils.data import Dataset, DataLoader

class LunarDatasetAlb(Dataset):
    def __init__(self, images_dir, masks_dir, augment=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images_list = sorted(os.listdir(images_dir))
        self.masks_list = sorted(os.listdir(masks_dir))
        self.augment = augment

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img = cv.imread(os.path.join(self.images_dir, self.images_list[idx]), cv.IMREAD_UNCHANGED)
        mask = cv.imread(os.path.join(self.masks_dir, self.masks_list[idx]), cv.IMREAD_UNCHANGED)

        # BGR->RGB; ensure mask is single-channel ints
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        if self.augment:
            out = self.augment(image=img, mask=mask)
            img_t = out["image"].float() / 255.0  # ToTensorV2 gives uint8 by default
            mask_t = out["mask"].long()
        else:
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask).long()

        return (img_t, mask_t)
    
    