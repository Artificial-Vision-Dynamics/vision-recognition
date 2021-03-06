import cv2
import time
import os
import numpy as np
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import pycocotools
import fiftyone as fo
import fiftyone.zoo as foz

## CV2 SALIENCY FUNCTIONS

folder = 'persons_val'
method = 2
numMin = 127
tipo = 4
## LOAD THE IMAGE (cambiar directory con lo que suba Rodrigo)
directory = "../" + folder
print(directory)
for i in range(58,59):
    f = os.listdir(directory)[i]
    nameImg = directory + '/' + f # str(i).zfill(12)
    print('Figure selected: ' + nameImg)
    
    img = cv2.imread(nameImg)
    #print('Valores de píxeles: '), print(img)
    #img = cv2.resize(img, (640, 480))
    #cv2.imshow('Visualizador OpenCV', img)
    
    ## COMPUTE SALIENCY MAP OF THE IMAGE (requires pip install opencv-contrib-python)
    init_time = time.time_ns() # Initialize time
    if method == 1:
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    if method == 2:
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()

    (success, saliencyMap) = saliency.computeSaliency(img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    #saliencyMap = img
    
    ## THRESHOLD OTSU
    if tipo == 1:
        threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2.imshow('Threshold OTSU', threshMap)
    
    ## THRESHOLD MAP
    if tipo == 2:
        ret, thresh_binary = cv2.threshold(saliencyMap, numMin, 255, cv2.THRESH_BINARY)
        ret, thresh_binary_inv = cv2.threshold(saliencyMap, numMin, 255, cv2.THRESH_BINARY_INV)
        ret, thresh_trunc = cv2.threshold(saliencyMap, numMin, 255, cv2.THRESH_TRUNC)
        ret, thresh_tozero = cv2.threshold(saliencyMap, numMin, 255, cv2.THRESH_TOZERO)
        ret, thresh_tozero_inv = cv2.threshold(saliencyMap, numMin, 255, cv2.THRESH_TOZERO_INV)
        # Displaying the different thresholding styles
        names = ['Original Image', 'Saliency map', 'BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV']
        images = img, saliencyMap, thresh_binary, thresh_binary_inv, thresh_trunc, thresh_tozero, thresh_tozero_inv
        # plt.subplot(1, 4, 1), plt.imshow(images[0]), plt.title(
        #     names[0]), plt.xticks([]), plt.yticks([])  # {1:2, 6:7}
        for i in range(0, 7):  # {2: 4, 6: 9}:  #
            plt1.subplot(2, 4, i+1), plt1.imshow(images[i], 'gray')
            plt1.title(names[i])
            plt1.xticks([]), plt1.yticks([])
        plt1.show()
    
    ## ADAPTATIVE THRESHOLDING
    if tipo == 3:
        ret, thresh_binary = cv2.threshold(saliencyMap, numMin, 255, cv2.THRESH_BINARY)
        thresh_mean = cv2.adaptiveThreshold(saliencyMap, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh_gaussian = cv2.adaptiveThreshold(saliencyMap, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        names = ['Original Image', 'Global Thresholding', 'Adaptive Mean Threshold', 'Adaptive Gaussian Thresholding']
        images = [img, thresh_binary, thresh_mean, thresh_gaussian]
        for i in range(4):
            plt2.subplot(2, 2, i+1), plt2.imshow(images[i], 'gray')
            plt2.title(names[i])
            plt2.xticks([]), plt2.yticks([])
        plt2.show()
        
    ## BITWISE OPERATION
    if tipo == 4:
        ret, mask = cv2.threshold(saliencyMap, numMin, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #apply AND operation on image and mask generated by thrresholding
        final = cv2.bitwise_and(img, img, mask=mask)
        #plot the result
        cv2.imshow('Bitwise operation', final)
    
    ## EDGE CALCULATION
    if tipo == 5:
        # Calculate the edges using Canny edge algorithm
        edges = cv2.Canny(img, 100, 200)
        # Plot the edges
        cv2.imshow('Canny edge algorithm',edges)
    
    end_time = time.time_ns()
    print('Algorithms duration: ', (end_time - init_time)/1e6, "ms")
    cv2.waitKey()
