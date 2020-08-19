



import os
import sys
import json
import datetime
import numpy as np

import glob
import skimage
from PIL import Image as pil_image

import cv2


import cv2

def locationToMask(locations=None,height=None,width=None):

    mask = np.zeros([height, width, len(locations)],
                    dtype=np.uint8)


    for index,location in enumerate(locations):
        x1, y1, x2, y2 = location
        mask[y1:y2+1,x1:x2+1,index]=1
        print(mask[:,:,index])







    return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)












def load_cmk(dataset_dir, subset):
    folder=os.path.join(dataset_dir, subset)
    imagesPattern=folder+'/*.jpg'

    for image_path in glob.glob(imagesPattern):
        print(image_path)

        img = cv2.imread(image_path)
        height,width = img.shape[:2]
        imageId=os.path.basename(image_path).replace('.jpg','')
        print(imageId)
        #
        # self.add_image(
        #     "balloon",
        #     image_id=a['filename'],  # use file name as a unique image id
        #     path=image_path,
        #     width=width, height=height,
        #     polygons=polygons)

        locationsFile='%s/%s.txt' % (folder,imageId)
        locations=[]
        with open(locationsFile) as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.replace('\n', '')
                if len(line.split(' ')) < 5:
                    break

                classIndex, xcen, ycen, w, h = line.strip().split(' ')
                xmin = max(float(xcen) - float(w) / 2, 0)
                xmax = min(float(xcen) + float(w) / 2, 1)
                ymin = max(float(ycen) - float(h) / 2, 0)
                ymax = min(float(ycen) + float(h) / 2, 1)

                xmin = int(width * xmin)
                xmax = int(width * xmax)
                ymin = int(height * ymin)
                ymax = int(height * ymax)

                location=(xmin,ymin,xmax,ymax)



                locations.append(location)
        print(locations)




















dataset_dir='/Volumes/v2/data/mlib_data/dataset/cmk/images_v2/'
subset='val'

load_cmk(dataset_dir=dataset_dir,subset=subset)

locations=[(2,3,5,7),(8,8,9,9)]
height=10
width=10
# mask,classIds=locationToMask(locations=locations,height=height,width=width)
# print(mask)
# print(classIds)

