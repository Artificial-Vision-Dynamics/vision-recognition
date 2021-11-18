import cv2
from matplotlib import pyplot as plt

img = cv2.imread('nadal.jpg', 0)
histg = cv2.calcHist([img], [0], None, [256], [0,256])

ret,th1 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

plt.imshow(th1, 'gray')
plt.show()

cv2.imwrite('nadal_150.jpg', th1)

