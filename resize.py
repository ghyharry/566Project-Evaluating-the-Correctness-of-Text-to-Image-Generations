import os
from PIL import Image
#import torch
#import torchvision
#import torchvision.transforms as T
#import cv2
#import numpy as np

#M = np.float32([[1, 0, 50], [0, 1, 50]])

def getAllFile(filename):
    goodSamples = os.walk(filename)
    
    files = []
    for _, _, fs in goodSamples:
        for f in fs:
            files.append(f)
    return files

def resizeAllFiles(filename, outputDir):
    files = getAllFile(filename)
    for i, file in enumerate(files):
        image = Image.open(filename + '/' + file)
        #image = cv2.imread(filename + '/' + file)
        #img_h = cv2.flip(image, 1)

        #(rows, cols) = img.shape[:2]
        #res = cv2.warpAffine(image, M, (cols, rows))

        #transform = T.RandomRotation((10, 90))
        new_image = image.resize((224,224))
        new_image.save(outputDir+ '/' + str(file.split('.')[0]) + '_224.jpg')

if __name__ == '__main__':
    resizeAllFiles('./nnsam', './jsam')