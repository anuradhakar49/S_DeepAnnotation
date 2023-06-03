# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2020. Authors: Cytomine SCRLFS.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */



# This is a sample of a software that can be run by the Cytomine platform using the Cytomine Python client (https://github.com/cytomine/Cytomine-python-client).
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from torch.utils.data import DataLoader

from utils.inference_dataloader import CustomDataset2
from natsort import natsorted
from glob import glob
#from unet import AttU_Net
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
########################################

import logging
import shutil

import cytomine
from cytomine.models import ImageInstanceCollection, JobData
from cytomine.models import Property, Annotation, AnnotationTerm, AnnotationCollection
from cytomine import Cytomine
#from cytomine.models import *

import string
from operator import sub
from cytomine.models import ProjectCollection
from cytomine.models import StorageCollection

import logging
from argparse import ArgumentParser

from shapely import wkt
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
from cytomine.models import Property, Project, Annotation, ImageInstance
from shapely.geometry import Polygon, LineString
from cytomine.models import AnnotationCollection, ImageInstanceCollection
from cytomine.models.ontology import Ontology, Term, RelationTerm, TermCollection

from cytomine.models import Project
from cytomine import Cytomine
from cytomine.models import CurrentUser


# -----------------------------------------------------------------------------------------------------------
def run(cyto_job, parameters):
    #logging.info("----- test software v%s -----", __version__)
    logging.info("Entering run(cyto_job=%s, parameters=%s)", cyto_job, parameters)

    job = cyto_job.job
    project = cyto_job.project ### project id provided

    # I create a working directory that I will delete at the end of this run
    working_path = os.path.join("tmp", str(job.id))
    if not os.path.exists(working_path):
        logging.info("Creating working directory: %s", working_path)
        os.makedirs(working_path)

    try:
        test_int_parameter = int(parameters.my_integer_parameter)

        logging.info("Display test_int_parameter %s", test_int_parameter)

        # loop for images in the project 
        images = ImageInstanceCollection().fetch_with_filter("project", project.id)
        nb_images = len(images)
        logging.info("# images in project: %d", nb_images)

        #value between 0 and 100 that represent the progress bar displayed in the UI.
        progress = 0
        progress_delta = 100 / nb_images
        
        ##########################################################@
        # code for running model, create annotations, upload annotations

        annotations = AnnotationCollection()
        main_data_dir = '/src/'
        MODEL_PATH = 'best_model_bg10.pth'
        
        #inf_dir = 'images' ## 132 slide
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
        _, _, files = next(os.walk("/images/"))
        print("number of files", len(files))
        inf_dir = '/images/'
        tmp_inf_imgs = natsorted(glob(os.path.join(inf_dir,'*.png')))  
        print("test point1", len(tmp_inf_imgs))
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
        #print("num images=", len(inf_loader))
        ct= 0
        all_poly= []
        for batch in tqdm(inf_loader):
            image = batch['image']
            print(image.shape)
            fname = tmp_inf_imgs[ct]
            fname = os.path.basename(os.path.normpath(fname))
            image = image.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                pred_mask = model(image)
                pred = torch.sigmoid(pred_mask)
                pred = (pred > 0.95).float()
                pred = pred.cpu()
            pred = Image.fromarray(np.uint8(np.squeeze(pred)*255))
            pred= np.asarray(pred)
            pred= cv2.dilate(pred, kernel, iterations=1)
            fname= fname.replace('.png','')

            #####################

            ps= 128
            x1, y1, x2, y2= float(fname.split('_')[1]), float(fname.split('_')[2]),float(fname.split('_')[3]),float(fname.split('_')[4])
            a= (x2-x1)/2
            b= (y2-y1)/2
            x3= x1 - 64+a  ##bottom left corner of patch
            y3= y1- 64+b
            x4= x3
            y4= y3-ps
            ret, thresh = cv2.threshold(pred, 127, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            polygons = []
            for object in contours:
                coordsx = []
                coordsy = []
                for point in object:
                    xval= float(point[0][0])+ float(x4)
                    yval=float(point[0][1])+ float(y4)+128
                    coordsx.append(xval) 
                    coordsy.append(yval)
                coords= list(zip(coordsx, coordsy))
                poly=np.array(coords)
                if len(poly)> 30:   
                    polygons.append(np.array(coords))
            all_poly.append(polygons) 
            ct+=1
        with Cytomine(host, public_key, private_key) as cytomine:
            me = CurrentUser().fetch()
            uname= str(me.username)
        id_image= 480387
        for i in range(1,len(all_poly)):
            if all_poly[i]!= []:
                polygon_patch1= Polygon(all_poly[i][0])
                polygon_patch1 = polygon_patch1.simplify(tolerance=simplify_tolerance)
                annotation_poly1 = Annotation(location=polygon_patch1.wkt, id_image=id_image).save()
                AnnotationTerm(annotation_poly1.id, id_term).save()









        #annotations.image = 10935422 
        #annotations.fetch()

        #l= len(annotations)
        #for annotation in annotations:
            #annotation.delete()
        ##############################################################


            

        l= len(all_poly)
        output_path = os.path.join(working_path, "num_annotations.txt")
        f= open(output_path,"w+")
        f.write("Number of annotations added %s\r\n" % str(l))
        #f.write("Number of annotations deleted %\r\n" % str(parameters.my_integer_parameter))
        f.close() 

        #I save a file generated by this run into a "job data" that will be available in the UI. 
        job_data = JobData(job.id, "Generated File", "num_annotations.txt").save()
        job_data.upload(output_path)

    finally:
        logging.info("Deleting folder %s", working_path)
        shutil.rmtree(working_path, ignore_errors=True)


        logging.debug("Leaving run()")


if __name__ == "__main__":
    logging.debug("Command: %s", sys.argv)

    with cytomine.CytomineJob.from_cli(sys.argv) as cyto_job:
        run(cyto_job, cyto_job.parameters)


