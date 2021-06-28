#!/usr/bin/env python

import numpy as np
import h5py
#import Image 
import os,sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.misc 
import scipy 
from scipy import misc 
from os    import path
from PIL import Image

image = Image.open("test.tiff") 
image = np.asarray(image)
size  = image.shape[0]

scan_side = size # Scan side in frames
nframes   = scan_side**2
[X,Y]     = np.meshgrid(range(0,scan_side),range(0,scan_side))

start_x = 0
start_y = 0
end_x = scan_side
end_y = scan_side

# Keep in mind that the translation refers to the sample moving, not the illumination!
translation = np.column_stack((X[start_y:end_y,start_x:end_x].flatten(),Y[start_y:end_y,start_x:end_x].flatten(),np.zeros(Y[start_y:end_y,start_x:end_x].size)))


step_size  = 1 # Scan step size in m
pixel_step = 1 # Step size in pixels
det_side   = size # Detector side in pixels

obj_side = size#det_side+(scan_side-1)*pixel_step

# Generate the probe
[X,Y] = np.meshgrid(np.arange(-3,3,6./det_side),np.arange(-3,3,6./det_side))
R = np.sqrt(X**2+Y**2)
# Use a relatively tight constraint
probe_mask = R < 0.3
probe = np.complex128(probe_mask)
probe *= np.random.random(probe.shape)
probe = np.fft.fftshift(np.fft.fft2(probe))

frames = np.empty((nframes,det_side,det_side))

for i in range(0,nframes):
    frames[i] = image[pixel_translation[i,1]:pixel_translation[i,1]+det_side,pixel_translation[i,0]:pixel_translation[i,0]+det_side]
    rframes[i] = frames[i]
    print('before frame * probe %d' %i)
    frames[i] = np.abs(np.fft.fftshift(np.fft.fft2(probe*frames[i])))**2