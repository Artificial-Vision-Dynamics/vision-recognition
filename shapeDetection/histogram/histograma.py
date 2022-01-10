import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('nadal.jpg', 0)
histg = cv2.calcHist([img], [0], None, [256], [0,256])

plt.plot(histg)
plt.show()

th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 12)

#plt.imshow(th2, 'gray')
#plt.show()
#th2d = cv2.bilateralFilter(th2, None, 15, 75)
th2d = (255-th2)
#kernel = np.ones((3,3), np.uint8)
#th2e = cv2.morphologyEx(th2d, cv2.MORPH_CLOSE, kernel, iterations=6)

kernel = np.ones((3,3), np.uint8)/9
th2e = cv2.filter2D(th2d, -1, kernel)

#plt.imshow(th2e, 'gray')
#plt.show()

#kernel = np.ones((2,2), np.uint8)
#th2e = cv2.morphologyEx(th2e, cv2.MORPH_OPEN, kernel, iterations=7)
kernel = np.ones((10,10), np.uint8)
th2e = cv2.morphologyEx(th2e, cv2.MORPH_CLOSE, kernel, iterations=3)

# Find outer contour and fill with white
cnts = cv2.findContours(th2e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cv2.fillPoly(th2e, cnts, [255,255,255])

kernel = np.ones((7,7), np.uint8)
th2e = cv2.morphologyEx(th2e, cv2.MORPH_OPEN, kernel, iterations=7)



#th2d = cv2.fastNlMeansDenoising(th2, None, 25)
#contours, hierarchy = cv2.findContours(th2d, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
#area = cv2.contourArea(contours)
#cv2.drawContours(img, contours, -1, (0,255,0), 3)


plt.imshow(th2e, 'gray')
plt.show()

cv2.imwrite('nadal_util.jpg', th2e)

prueba_mask = cv2.bitwise_and(img, img, mask=th2e)
plt.imshow(prueba_mask, 'gray')
plt.show()
cv2.imwrite('nadal_mask.jpg', prueba_mask)
