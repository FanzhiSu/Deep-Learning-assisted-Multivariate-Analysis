# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 11:17:26 2022

@author: Clark
"""

import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision.io import read_image

class pytorch_data(Dataset):
    
    def __init__(self, label, data_dir, transform):             
        # Get Labels
        self.labels = pd.read_csv(label)       
        #Get image
        self.image_dir = data_dir       
        self.transform = transform      
    def __len__(self):
        return len(self.labels) # size of dataset
      
    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        img_path = os.path.join(self.image_dir, self.labels.iloc[idx, 0])
        # image = read_image (img_path)                       
        image = Image.open(img_path)  # Open Image with PIL
        image = np.array (image)
        image = self.transform(image) # Apply Specific Transformation to Image
        labels = self.labels.iloc[idx, 1]
        labels = int(labels)
        return image.float(), labels





