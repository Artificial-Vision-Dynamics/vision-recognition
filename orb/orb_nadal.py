import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('orb/nadal_mask.jpg',0)

orb = cv.ORB_create()
kp = orb.detect(img, None)
kp, des = orb.compute(img, kp)

img2 = cv.drawKeypoints(img, kp, None, color=(0,250,0), flags=0)
plt.imshow(img2)
plt.show()
cv.imwrite('orb/orb_keypoints.jpg',img2)



sift = cv.SIFT_create()
kp2 = sift.detect(img,None)
img3=cv.drawKeypoints(img,kp,None)
plt.imshow(img3)
plt.show()
cv.imwrite('orb/sift_keypoints.jpg',img3)
