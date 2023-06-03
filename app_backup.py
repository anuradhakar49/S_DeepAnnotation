#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from torch.utils.data import DataLoader

from utils import CustomDataset2, confusion_matrix, dice_coeff_batch, DiceLoss
from natsort import natsorted
from glob import glob
from unet import AttU_Net
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import os, sys
import random
import torch
import time
import cv2

from PIL import Image
import matplotlib.pyplot as plt


from glob import glob
from unet.unet_model import AttU_Net

main_data_dir = '/Users/anuradha.kar/Documents/python_scripts/docker_tests/docker_py1/src/'
MODEL_PATH = 'best_model_bg10.pth'

inf_dir = 'images' ## 132 slide
pin_memory= True
scale_factor = 1.0
n_input_channels = 3
inf_imgs_fold = []
batch_size = 1
n_devices = 1
n_workers = 4

n_input_channels = 3

n_output_channels = 1
kernel = np.ones((3, 3), np.uint8)

if __name__ == "__main__":  
 
    tmp_inf_imgs = natsorted(glob(os.path.join(main_data_dir,inf_dir,'*.png')))
    for s in range(len(tmp_inf_imgs)):
        inf_imgs_fold.append(tmp_inf_imgs[s])
    inf_dataset = CustomDataset2(inf_imgs_fold, normalize=False,cached_data=pin_memory, n_channels=n_input_channels,scale=scale_factor)
    inf_loader = DataLoader(inf_dataset, batch_size=batch_size*n_devices, shuffle=False, pin_memory=False, num_workers=n_workers)
    model = AttU_Net(img_ch=n_input_channels, output_ch=n_output_channels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    model.to(device=device, dtype=torch.float32)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    ct= 0
    for batch in tqdm(inf_loader):
        image = batch['image']
        fname = tmp_inf_imgs[ct]
        fname = os.path.basename(os.path.normpath(fname))
        
        image = image.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            pred_mask = model(image)
            pred = torch.sigmoid(pred_mask)
            pred = (pred > 0.95).float()
            pred = pred.cpu()
            print(pred.shape)
        pred = Image.fromarray(np.uint8(np.squeeze(pred)*255))

        #pred.save("/src/outputs/pred.jpg")
        pred= np.asarray(pred)
        pred= cv2.dilate(pred, kernel, iterations=1)
        cv2.imwrite('/src/outputs/'+ fname, pred)
        ct+=1
 





