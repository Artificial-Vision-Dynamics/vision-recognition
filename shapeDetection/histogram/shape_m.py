import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def shape(gray):
    ret3,th3 = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    th3 = cv.morphologyEx(th3, cv.MORPH_CLOSE, kernel, iterations=5)
    kernel = np.ones((7,7), np.uint8)
    th3 = cv.morphologyEx(th3, cv.MORPH_OPEN, kernel, iterations=15)
    plt.imshow(th3,'gray')
    plt.show()
    mask = cv.inRange(th3, 0, 0)
    cnts = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    hwmax = 0
    cmax = 0
    for c in cnts:
        x,y,w,h = cv.boundingRect(c)
        hw = h*w
        if hw > hwmax:
            x1,y1,w1,h1 = x,y,w,h
            hwmax = h*w
            cmax = c
    return x1,y1,w1,h1


name = 'daniel'
img_rgb = cv.imread(name+'.jpg')
img = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
h = len(img)
w = len(img[0])

x0,y0,w0,h0 = shape(img)
img0 = img[y0:y0+h0, x0:x0+w0]
# plt.imshow(img1,'gray')
# plt.show()

x1,y1,w1,h1 = shape(img0)
img1 = img0[y1:y1+h1, x1:x1+w1]

ret3,th3 = cv.threshold(img1,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
kernel = np.ones((7,7), np.uint8)
closing = cv.morphologyEx(th3, cv.MORPH_CLOSE, kernel,iterations=5)
kernel = np.ones((15,15), np.uint8)
opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel, iterations=25)
erosion = cv.erode(opening,kernel,iterations = 10)
# plt.imshow(erosion,'gray')
# plt.show()

filtro = 255*np.ones((1,w1), dtype=int)
filtro1 = 255*np.ones((h0,1), dtype=int)

# print(len(img1),len(img1[0]))

# print(h,w)
# print(x0,y0)
# print(h0,w0)
# print(x1,y1)
# print(h1,w1)
# print("")

# print(len(erosion),len(erosion[0]))

for k in range(h0):  
    if k+1 < y1:
        filtro=np.append(filtro, 255*np.ones((1,w1), dtype=int),axis=0)
    elif k == y1:
        if y1 == 0:
            filtro = erosion
        else:
            filtro=np.append(filtro, erosion, axis = 0)
    elif k >= y1+h1:
        filtro=np.append(filtro, 255*np.ones((1,w1), dtype=int),axis=0)
        
# print(len(filtro),len(filtro[0]))        
# plt.imshow(filtro, 'gray')
# plt.show()

for j in range(w0):
    if j+1<x1:
        filtro1 = np.append(filtro1,255*np.ones((h0,1), dtype=int),axis = 1)
    elif j == x1:
        if x1 == 0:
            filtro1 = filtro
        else:
            filtro1 = np.append(filtro1, filtro, axis = 1)
    elif j >= x1+w1:
        filtro1 = np.append(filtro1,255*np.ones((h0,1), dtype=int),axis = 1)
# plt.imshow(filtro1, 'gray')
# plt.show()

# print(len(filtro1),len(filtro1[0])) 
# plt.imshow(filtro1,'gray')
# plt.show()
# print(h0,w0)
# print(len(filtro1),len(filtro1[0]))

filtro2 = 255*np.ones((1,w0), dtype=int)
filtro3 = 255*np.ones((h,1), dtype=int)

# print(len(img1),len(img1[0]))

# print(y1, x1)

for m in range(h):  
    if m+1 < y0:
        filtro2 = np.append(filtro2, 255*np.ones((1,w0), dtype=int),axis=0)
    elif m == y0:
        if y0 == 0:
            filtro2 = filtro1
        else:
            filtro2 = np.append(filtro2, filtro1, axis = 0)
    elif m >= y0+h0:
        filtro2 = np.append(filtro2, 255*np.ones((1,w0), dtype=int),axis=0)
        
# print(len(filtro2),len(filtro2[0]))        
# plt.imshow(filtro2, 'gray')
# plt.show()

for n in range(w):
    if n+1<x0:
        filtro3 = np.append(filtro3,255*np.ones((h,1), dtype=int),axis = 1)
    elif n == x0:
        if x0 == 0:
            filtro3 = filtro2
        else:
            filtro3 = np.append(filtro3, filtro2, axis = 1)
    elif n >= x0+w0:
        filtro3 = np.append(filtro3,255*np.ones((h,1), dtype=int),axis = 1)
# plt.imshow(filtro1, 'gray')
# print(len(filtro3),len(filtro3[0]))   

mask = cv.inRange(filtro3, 0, 0)           
res = cv.bitwise_and(img_rgb, img_rgb, mask=mask)
plt.imshow(res,'gray')
plt.show()
cv.imwrite(name+'_mask.jpg', res)
cv.imwrite(name+'_util.jpg',filtro3)