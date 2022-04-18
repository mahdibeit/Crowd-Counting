import numpy as np
import pandas as pd
import scipy.io
import cv2
from pathlib import Path
import json 
import fnmatch
import os
import glob

url = r"/content/drive/MyDrive/crowd-counting/dataset/shanghai/ShanghaiTech/part_A/asli/Train/"
url_ =r"/content/drive/MyDrive/crowd-counting/dataset/shanghai/ShanghaiTech/part_A/asli/"


data = glob.glob("/content/drive/MyDrive/crowd-counting/dataset/shanghai/ShanghaiTech/part_A/asli/Train/*.jpg")


image_ad = []
label_ad = []
image = []
label = []
labels = []
counts = []
locs = []
img = []
ws = []
hs = []

for i in range(len(data)):
  file_name = Path(data[i]).stem
  print(file_name)

  image_ad.append(file_name)
  label_ad.append(file_name)

  image.append(url + file_name +'.jpg')
  label.append(url + file_name +'.mat')

  labels.append(scipy.io.loadmat(label[i]))

  counts.append(labels[i]['image_info'][0][0][0][0][1][0][0])
  # print('counts:', counts[i])
  locs.append(labels[i]['image_info'][0][0][0][0][0])
  # print('locs:', locs[i])
  img.append(cv2.imread(image[i]))

  ws.append(img[i].shape[0])
  hs.append(img[i].shape[1])

  weights = [0, -0.5, 1, 0.5 ]
  patches = []
  psizes = []
  #plabels = []
  for k in range(len(weights)):
      if k == 0 or k== 1:
        patches.append(img[i][int(abs(ws[i]*weights[k])):int(ws[i]*(1/2-weights[k])),int(abs(hs[i]*weights[k])):int(hs[i]*(1/2-weights[k]))])
        psizes.append([int(abs(ws[i]*weights[k])),int(ws[i]*(1/2-weights[k])), int(abs(hs[i]*weights[k])), int(hs[i]*(1/2-weights[k]))])
      elif k == 2 : 
        patches.append(img[i][int(ws[i]*(weights[k]-1/2)):int(ws[i]*weights[k]),int(hs[i]*(1-weights[k])):int(hs[i]*(-1/2 + weights[k]))])
        psizes.append([int(ws[i]*(weights[k]-1/2)),int(ws[i]*weights[k]),int(hs[i]*(1-weights[k])),int(hs[i]*(-1/2 + weights[k]))])
      else:
        patches.append(img[i][int(ws[i]*(weights[k]-1/2)):int(ws[i]*weights[k]),int(hs[i]*(1-weights[k])):int(hs[i]*(1/2 + weights[k]))])
        psizes.append([int(ws[i]*(weights[k]-1/2)),int(ws[i]*weights[k]),int(hs[i]*(1-weights[k])),int(hs[i]*(1/2 + weights[k]))])
  count = 0
  print(psizes)
  for j in range(len(psizes)):
      temp = []
      for [x,y] in locs[i]:
          if (y <= psizes[j][1]) and (y >= psizes[j][0]) and (x <= psizes[j][3]) and (x >= psizes[j][2]):
            if count == 0:
                temp.append([x,y])
            elif count == 1:

                temp.append([x - (psizes[j][3]- psizes[j][2]) ,y- (psizes[j][1]- psizes[j][0])])       
            elif count == 2:
                temp.append([x,y- (psizes[j][1]-psizes[j][0])])
            else:
                temp.append([x - (psizes[j][3]- psizes[j][2]),y])

      count+=1
      print('len temp: ', len(temp))
      if len(temp)> 0:
        cv2.imwrite(url_ + "patch_data/Train/" + image_ad[i] + "_patch_" + str(j) + ".jpg", patches[j])
        patchlabel = labels[i].copy()
        patchlabel['image_info'][0][0][0][0][1][0][0] = len(temp)
        # count += len(temp)
        patchlabel['image_info'][0][0][0][0][0] = np.array(temp)
        scipy.io.savemat(url_ + "patch_data/Train/" + label_ad[i] + "_patch_" + str(j) + ".mat", patchlabel)
  # print("IMAGE " + image_ad[i] + " Count:",count)

