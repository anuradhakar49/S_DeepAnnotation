#!/usr/bin/env python
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from torch.utils.data import DataLoader

from utils import CustomDataset2, confusion_matrix, dice_coeff_batch, DiceLoss
from natsort import natsorted
from glob import glob
from unet import AttU_Net
from tqdm import tqdm
import numpy as np
import logging
import os, sys
import random
import torch
import time
import cv2

from PIL import Image


from glob import glob
from unet.unet_model import AttU_Net

############ cytomine libraries

from cytomine import Cytomine
#from cytomine.models import *
import numpy as np
#from skimage import io
import string
from operator import sub
import os

from cytomine.models import ProjectCollection
#from cytomine.models import AnnotationCollection, ImageInstanceCollection
from cytomine.models import StorageCollection

import logging
import sys
from argparse import ArgumentParser

import os

from shapely import wkt
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
#from cytomine.models import AnnotationCollection, ImageInstanceCollection
from cytomine.models import Property, Project, Annotation, ImageInstance
from cytomine.models import Property, Annotation, AnnotationTerm, AnnotationCollection
from shapely.geometry import Polygon, LineString
from cytomine.models import AnnotationCollection, ImageInstanceCollection
from cytomine.models.ontology import Ontology, Term, RelationTerm, TermCollection

#######################
from cytomine.models import Project
from cytomine import Cytomine
from cytomine.models import CurrentUser

########################@



main_data_dir = '/Users/anuradha.kar/Documents/python_scripts/docker_tests/docker_py1/src/'
MODEL_PATH = 'best_model_bg10.pth'


inf_dir = 'images' ## 132 slide
pin_memory= True
scale_factor = 1.0
n_input_channels = 3
inf_imgs_fold = []
batch_size = 1
n_devices = 1
n_workers = 0

n_input_channels = 3

n_output_channels = 1
kernel = np.ones((3, 3), np.uint8)
######################################################

id_image=480387 ## fixed for admin acc image
id_term= 360612
simplify_tolerance = 5
######################################################

host = 'https://cytomine-staging.icm-institute.org'#https://cytomine-staging.icm-institute.org'#'https://demo.cytomine.com'
public_key = 'a40ce3ac-a46c-4018-8d29-a3906121a9c9' 
private_key = 'c662cb38-95f3-4fbd-9253-3b79a90667ce'
#######################################################

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
    all_poly= []
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
            #print(pred.shape)
        pred = Image.fromarray(np.uint8(np.squeeze(pred)*255))

        #pred.save("/src/outputs/pred.jpg")
        pred= np.asarray(pred)
        pred= cv2.dilate(pred, kernel, iterations=1)
        fname= fname.replace('.png','')
        ##### annotation creation
        ps= 128
        x1, y1, x2, y2= float(fname.split('_')[1]), float(fname.split('_')[2]),float(fname.split('_')[3]),float(fname.split('_')[4])
        a= (x2-x1)/2
        b= (y2-y1)/2
        x3= x1 - 64+a  ##bottom left corner of patch
        y3= y1- 64+b
        x4= x3  ## top left corner which is the origin (0,0) for the patch coordinates
        y4= y3-ps
        ret, thresh = cv2.threshold(pred, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for object in contours:
            coordsx = []
            coordsy = []
            for point in object:
                xval= float(point[0][0])+ float(x4)
                yval=float(point[0][1])+ float(y4)+128 ##128 added for shift
                coordsx.append(xval) ## add x directly here
                coordsy.append(yval) ## add y directly here
            coords= list(zip(coordsx, coordsy))
            poly=np.array(coords)
            if len(poly)> 30:   
                polygons.append(np.array(coords))
        all_poly.append(polygons) 
        ct+=1

    with Cytomine(host, public_key, private_key) as cytomine:
        me = CurrentUser().fetch()
        uname= str(me.username)

    for i in range(1,len(all_poly)):
        if all_poly[i]!= []:
            polygon_patch1= Polygon(all_poly[i][0])
            polygon_patch1 = polygon_patch1.simplify(tolerance=simplify_tolerance)
            annotation_poly1 = Annotation(location=polygon_patch1.wkt, id_image=id_image).save()
            AnnotationTerm(annotation_poly1.id, id_term).save()
 





