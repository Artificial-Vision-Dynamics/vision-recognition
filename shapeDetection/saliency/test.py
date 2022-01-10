import cv2
import time


img = cv2.imread('nadal.jpg')
img = cv2.resize(img, (640, 480))
cv2.imshow('Visualizador OpenCV', img)

init_time = time.time_ns()

# requires pip install opencv-contrib-python
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
#saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(img)
saliencyMap = (saliencyMap * 255).astype("uint8")

# create binary img
threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# draw contours
contours, hierarchy = cv2.findContours(threshMap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_cont = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
end_time = time.time_ns()

print('Algorithm duration: ', (end_time - init_time)/1e6, "ms")

# imshow
cv2.imshow('Binary SaliencyMap', threshMap)
cv2.imshow('SaliencyMap', saliencyMap)
cv2.imshow('Contours', img_cont)
cv2.waitKey()