# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 17:54:55 2020

@author: klal1
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def format_color_image(img):
    img = img.convert('RGB')
    return np.array(img.resize((768, 1152), Image.ANTIALIAS), 'float32')/127.5 - 1

def format_black_image(img):
    return np.array(img.resize((768, 1152), Image.ANTIALIAS), 'float32').reshape(1152,768,1)/127.5-1

def get_images(colorImagePath, greyImagePath):
    images = {}
    i, j = 0, 0
    for fileName in os.listdir(colorImagePath):
        file = os.path.join(colorImagePath,fileName)
        img = Image.open(file)
        if(img.mode == 'RGB'):
            images[fileName[:-4]] = [np.array(img.resize((768, 1152), Image.ANTIALIAS), 'float32')/127.5 - 1]
            i += 1
            
    for fileName in os.listdir(greyImagePath):
        file = os.path.join(greyImagePath, fileName)
        img = Image.open(file)
        if(fileName[:-4] in images):
            images[fileName[:-4]].append(np.array(img.resize((768, 1152), Image.ANTIALIAS), 'float32').reshape(1152,768,1)/127.5-1)
            j += 1
    print(i, j)
    return images

def data_gen(Set, batch_size):
    while(True):
        keys = np.random.choice(list(Set.keys()), batch_size)
        gImg, cImg = [], []
        for k in keys:
            cImg.append(Set[k][0])
            gImg.append(Set[k][1])
        yield np.array(gImg), np.array(cImg)
        
        
        
def get_image_set(colorPath, blackPath, names):
    gList = []
    cList = []
    for name in names:
        imCpath = os.path.join(colorPath,name)
        imGpath = os.path.join(blackPath,name)
        imgC = Image.open(imCpath)
        imgG = Image.open(imGpath)
        gList.append(format_black_image(imgG))
        cList.append(format_color_image(imgC))
    return np.array(gList), np.array(cList)
  
def save_and_plot_image(t_cimg, t_gimg, gen, path):
    h, w, c = t_cimg.shape
    fake_A = gen.predict(t_gimg.reshape(1,1152,768,1))
    fake_A = 0.5 * fake_A + 0.5
    plt.imsave(path, fake_A.reshape(h, w, c))