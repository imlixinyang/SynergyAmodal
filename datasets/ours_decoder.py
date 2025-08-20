# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torchvision.transforms.v2 as transforms

to_tensor = transforms.ToTensor()
import pycocotools.mask as maskUtils

import torch
import torch.utils.data
import torch.nn.functional as F

import json

import numpy as np
from PIL import Image
import os
import random

def mask_to_bbox(mask):
    mask = (mask == 1)
    if np.all(~mask):
        return [0, 0, 0, 0]
    assert len(mask.shape) == 2
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [cmin.item(), rmin.item(), cmax.item() + 1 - cmin.item(), rmax.item() + 1 - rmin.item()] # xywh

class OurPseudoDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, amodal_dir, amodal_annotations_path, image_size=512, length_multiplier=1):
        super().__init__()
        
        self.image_dir = image_dir
        self.amodal_dir = amodal_dir
        self.amodal_annotations_path = amodal_annotations_path
        
        with open(amodal_annotations_path, 'r') as f:
            self.amodal_annotations = json.load(f)
        
        self.length_multiplier = length_multiplier
        
        self.real_length = len(self.amodal_annotations)
        
        print(f"Dataset length: {self.real_length}")
        
        self.image_size = image_size

        self.transforms = transforms.Compose(
            [
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.ToTensor(),
            ]
        )   
        
    def __len__(self):
        return self.real_length * self.length_multiplier
        
    def collate_fn(self, batch):
        
        return {
                'amodal_image': torch.stack([sample['amodal_image'] for sample in batch]),
                'amodal_mask': torch.stack([sample['amodal_mask'] for sample in batch]),
            }
                
    def __getitem__(self, idx):
        
        try:
        # if True:
            idx = idx % self.real_length

            annotation = self.amodal_annotations[idx]
            
            image_path = os.path.join(self.image_dir, annotation['filename'])
            image = to_tensor(Image.open(image_path).convert("RGB"))

            H, W = image.shape[1:]

            if 'mask' in annotation:
                by0, bx0, w, h = annotation['bbox']
                cropped_modal_mask =  torch.from_numpy(maskUtils.decode(annotation['mask']))

                modal_mask = torch.zeros(H, W)
                modal_mask[bx0:bx0+h, by0:by0+w] = cropped_modal_mask

                amodal_mask = modal_mask.clone()

                amodal_image = image.clone() * modal_mask[None] + 1 * (1 - modal_mask[None])
            else:
                by0, bx0, w, h = annotation['bbox']
                cropped_modal_mask = torch.from_numpy(maskUtils.decode(annotation['modal_mask']))

                modal_mask = torch.zeros(H, W)
                modal_mask[bx0:bx0+h, by0:by0+w] = cropped_modal_mask

                amodal_mask = torch.zeros(H, W)
                cropped_amodal_mask = torch.from_numpy(maskUtils.decode(annotation['amodal_mask']))
                amodal_mask[bx0:bx0+h, by0:by0+w] = cropped_amodal_mask

                cropped_amodal_image = to_tensor(Image.open(os.path.join(self.amodal_dir, annotation['amodal_rgb'])).convert("RGB"))

                amodal_image = torch.ones(3, H, W)
                amodal_image[:, bx0:bx0+h, by0:by0+w] = cropped_amodal_image * cropped_amodal_mask[None] + 1 * (1 - cropped_amodal_mask[None])

            all = torch.cat([image, modal_mask[None], amodal_image, amodal_mask[None]], dim=0)

            if random.random() < 0.7:
                if min(H, W) < self.image_size:
                    all = F.interpolate(all[None], size=(int(H * self.image_size / min(H, W)), int(W * self.image_size / min(H, W))), mode='bilinear', align_corners=False)[0]
                else:
                    max_scale = 1
                    min_scale = self.image_size / min(H, W)

                    if np.random.rand() < 0.7:
                        random_scale = np.exp(np.random.rand() * (np.log(max_scale) - np.log(min_scale)) + np.log(min_scale))
                    else:
                        random_scale = self.image_size / min(H, W)

                    all = F.interpolate(all[None], size=(int(H * random_scale), int(W * random_scale)), mode='bilinear', align_corners=False)[0]

                # augment and crop
                num_tries = 0
                while True:
                    augmented_all = self.transforms(all)
                    if augmented_all[3].sum() > 16 * 16:
                        break
                    num_tries += 1
                    if num_tries > 10:
                        raise Exception(f"Failed to get a valid sample at {idx}")
                    
                H = self.image_size
                W = self.image_size
            else:
                max_scale = self.image_size / min(H, W)
                min_scale = self.image_size / max(H, W)

                if np.random.rand() < 0.7:
                    random_scale = np.exp(np.random.rand() * (np.log(max_scale) - np.log(min_scale)) + np.log(min_scale))
                else:
                    random_scale = self.image_size / max(H, W)

                all = F.interpolate(all[None], size=(int(H * random_scale), int(W * random_scale)), mode='bilinear', align_corners=False)[0]

                H, W = all.shape[1:]

                if H > self.image_size:
                    random_crop_H = np.random.randint(0, H - self.image_size)
                    all = all[:, random_crop_H:random_crop_H + self.image_size, :]
                    H = self.image_size

                if W > self.image_size:
                    random_crop_W = np.random.randint(0, W - self.image_size)
                    all = all[:, :, random_crop_W:random_crop_W + self.image_size]
                    W = self.image_size

                augmented_all = F.pad(all, (0, self.image_size - W, 0, self.image_size - H), value=0)

            image, modal_mask, amodal_image, amodal_mask = augmented_all.split([3, 1, 3, 1], dim=0)  

            modal_mask = (modal_mask > 0.5).float()
            amodal_mask = (amodal_mask > 0.5).float()

            amodal_image[:, H:, :] = 1
            amodal_image[:, :, W:] = 1
            
            return {
                    'amodal_image': amodal_image,
                    'amodal_mask': amodal_mask,
            }
        
        except:
            # print(f"Failed to get a valid sample at {idx}")
            return self.__getitem__((idx+1) % self.real_length) 
            

# unit test
if __name__ == "__main__":
    from torchvision.utils import save_image
    dataset = OurPseudoDataset(
        image_dir="SynergyAmodal16K/images", 
        amodal_dir="SynergyAmodal16K/amodal_rgbs", 
        amodal_annotations_path="SynergyAmodal16K/amodal_annotations.json", 
        image_size=512, 
        length_multiplier=1)

    os.makedirs("data/ours_tmp", exist_ok=True)
    
    for i in range(100):
        if True:
            data = dataset[i]
            amodal_image = data['amodal_image']
            amodal_mask = data['amodal_mask']
            save_image(amodal_image, f"data/ours_tmp/{i}_amodal_image.png")
            save_image(amodal_mask, f"data/ours_tmp/{i}_amodal_mask.png")