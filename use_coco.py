from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm
import requests

# it is important that it is only for train/val set of imgs
# this is the folder where I am running everything, you should change it
dataDir= 'opencv/'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# create coco da
coco = COCO(annFile)

categories = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in categories]
#print('COCO categories: \n{}\n'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds) # it returns all imgs ids
print('Img dataset of people: ', len(imgIds))

img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

filename = dataDir + dataType + '/' + img['file_name']
print(filename)
img_cv2 = cv2.imread(filename)
img_cv2 = img_cv2[:,:,(2,1,0)]
plt.axis('off')
plt.imshow(img_cv2)
plt.show()

annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
coco_kps=COCO(annFile)
plt.imshow(img_cv2)
plt.axis('off')
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco_kps.loadAnns(annIds)
coco_kps.showAnns(anns)
plt.show()


# this part is for saving iimages
#print("Saving the images with required categories ...")
#imgs_dir = dataDir + 'persons_val'
#os.makedirs(imgs_dir, exist_ok=True)
#images = coco.loadImgs(imgIds)
# Save the images into a local folder
#for im in tqdm(images):
#    img_data = requests.get(im['coco_url']).content
#    with open(os.path.join(imgs_dir, im['file_name']), 'wb') as handler:
#        handler.write(img_data)