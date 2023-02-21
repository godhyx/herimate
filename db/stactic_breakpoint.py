import os
import json
import shutil
import random
import cv2
import numpy as np

ANNO_PATH = '/home/ps/Data/LD_VIT/MLDS_TrainingDataSet_v0.2.0/'
new_anno_path = '/home/ps/Data/LD_VIT/MLDS_TrainingDataSet_v0.2.0_1/'
if not os.path.exists(new_anno_path):
    os.makedirs(new_anno_path)

files = os.listdir(ANNO_PATH)
all_num = 0
breakpoint_num = 0
for file in files:
    if file.endswith('json'):
        filePath = ANNO_PATH + file
        with open(filePath, 'r', encoding='utf-8') as fr:
            content = json.load(fr)
        all_num += 1
        dataList = content['dataList']
        for j in range(len(dataList)):
            properties = dataList[j]['properties']
            if len(properties)>0:
                groupId = properties['groupId']
                if groupId>=0:
                    breakpoint_num += 1
                    shutil.copy(filePath, new_anno_path+file)
                    shutil.copy(ANNO_PATH+file[:-4]+'jpg', new_anno_path + file[:-4]+'jpg')
                    break
print(all_num, breakpoint_num, breakpoint_num/all_num)
        
# print(i)

