#!/usr/bin/env python

import numpy as np
import os,sys
import scipy 
from os import path
from PIL import Image
import torch 


probe = np.load('probe.npy',allow_pickle=True) 
size  = probe.shape[0]

image = Image.open("persian_motif.jpg") 
image = np.asarray(image)
image = np.resize(image,size)

scan_side = size # Scan side in frames
nframes   = scan_side**2
[X,Y]     = np.meshgrid(range(0,scan_side),range(0,scan_side))

start_x = 0
start_y = 0
end_x = scan_side
end_y = scan_side

# Keep in mind that the translation refers to the sample moving, not the illumination!
translation = np.column_stack((X[start_y:end_y,start_x:end_x].flatten(),Y[start_y:end_y,start_x:end_x].flatten(),np.zeros(Y[start_y:end_y,start_x:end_x].size)))


step_size  = 1    # Scan step size in m
pixel_step = 1    # Step size in pixels
det_side   = size # Detector side in pixels

obj_side = size

# Generate the probe
[X,Y] = np.meshgrid(np.arange(-3,3,6./det_side),np.arange(-3,3,6./det_side))
R = np.sqrt(X**2+Y**2)

pixel_translation = translation * pixel_step
# Use a relatively tight constraint
#probe = np.complex128(probe_mask)
#probe *= np.random.random(probe.shape)
#probe = np.fft.fftshift(np.fft.fft2(probe))

frames = np.empty((nframes,det_side,det_side))

for i in range(0,nframes):
    object = image[pixel_translation[i,1]:pixel_translation[i,1]+det_side,pixel_translation[i,0]:pixel_translation[i,0]+det_side]
    frames.append(np.abs(np.fft.fftshift(np.fft.fft2(probe*object)))**2)
frames = np.array(frames)

np.save('frames.npy',frames,allow_pickle=True)
