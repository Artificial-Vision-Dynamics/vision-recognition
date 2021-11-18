import cv2
from matplotlib import pyplot as plt

img = cv2.imread('nadal.jpg', 0)
histg = cv2.calcHist([img], [0], None, [256], [0,256])

th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 5)

plt.imshow(th2, 'gray')
plt.show()

cv2.imwrite('nadal_2_15.jpg', th2)

