from itertools import count
import os
import json
import shutil
import random
from xml.dom.minidom import Identified
import cv2
import numpy as np

#FILE_PATH = '/home/bess/DISK1/FODNet_dataset/OD_train_city_0.0.7/test_data/'  #'./rgb_images'
#ANNO_PATH = '/home/vdb/SurroundViewData/04DataSet/TrainingDataSet/MLDS_TrainingDataSet/MLDS_TrainingDataSet_v0.2.6_7426/yongdu/'
ANNO_PATH = '/home/gy/Data/LD_VIT/order/'

# NEW_FILE_PATH = '/home/bess/DISK1/FODNet_dataset/OD_train_city_0.0.7_1280/test_data/'
NEW_ANNO_PATH = '/home/gy/Data/LD_VIT/order_ok/'
if not os.path.exists(NEW_ANNO_PATH):
    os.makedirs(NEW_ANNO_PATH)

labels = []
IMG_HEIGHT = 192 #360  #1080
IMG_WIDTH = 960  #640  #1920

files = os.listdir(ANNO_PATH)
i = 0
for file in files:
    if file.endswith('json'):
        print(file)
        # file = '20210619_080041_0026_00_000262144_260133540.MP4_2820.json'
        img_path  = os.path.join(ANNO_PATH, file[:-4]+'jpg')
        img = cv2.imread(img_path)
        r = IMG_WIDTH / img.shape[1]
        # assert r == 0.5 
        filePath = ANNO_PATH + file
        with open(filePath, 'r', encoding='utf-8') as fr:
            content = json.load(fr)

        if 'skyline' in content:
            skyline = content['skyline']
            randline = random.randint(-10,10)
            crop_a = int(skyline-50/r+randline)
            crop_b = int((img.shape[0] - IMG_HEIGHT/r)-crop_a)
        # min_y = 10000
        # if 'shapes' in content:
        #     shapes = content['shapes']
        #     for shape in shapes:
        #         org_shape=[]
        #         if shape['points'][0][1] < shape['points'][-1][1]:
        #             for i in range(len(shape['points'])-1, -1, -1):
        #                 org_shape.append(shape['points'][i])
        #         else:
        #             org_shape = shape['points']
        #         if min_y>org_shape[-1][1]:
        #             min_y = org_shape[-1][1]
        # if min_y != 10000:
        #     crop_a = int(min_y-100)
        #     crop_b = int((img.shape[0] - IMG_HEIGHT/r)-crop_a)
        else:
            crop_a = int((img.shape[0] - IMG_HEIGHT/r)*0.75)
            crop_b = int((img.shape[0] - IMG_HEIGHT/r)*0.25)
        # print(img.shape[0], IMG_HEIGHT*r, crop_a, crop_b)
        img = img[crop_a:img.shape[0]-crop_b,:]
        # print(img.shape)
        
        assert IMG_HEIGHT / img.shape[0] == IMG_WIDTH / img.shape[1]
        new_img_path = os.path.join(NEW_ANNO_PATH, file[:-4]+'jpg')
        out_anno_path = NEW_ANNO_PATH + file
        if r == 1:
            # print(r)
            shutil.copy(img_path, new_img_path)
            shutil.copy(ANNO_PATH + file, out_anno_path)
        else:
            resized_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT),
                interpolation=cv2.INTER_LINEAR,).astype(np.uint8)

            if 'shapes' in content:
                shapes = content['shapes']
                for ind in range(len(shapes)):
                    shape = shapes[ind]
                    for id in range(len(shape['points'])):
                        shape['points'][id][0] = shape['points'][id][0] * r
                        shape['points'][id][1] = (shape['points'][id][1]-crop_a) * r
                
                for i in range(len(shapes)):
                    org_shape = shapes[i]
                    groupId = org_shape['group_id']
                    final_point = org_shape['points'][-1][1]
                    if groupId:
                        for j in range(i+1,len(shapes)):
                            new_shape = shapes[j]
                            if groupId == new_shape['group_id']:
                                if new_shape['points'][-1][1] > final_point:
                                    temp = shapes[i]
                                    shapes[i] = shapes[j]
                                    shapes[j] = temp
                            else:
                                continue
                
                content['shapes'] = shapes
            with open(out_anno_path, 'w', encoding='utf-8') as fw:
                json.dump(content,fw,ensure_ascii=False,indent=2)
            cv2.imwrite(new_img_path, resized_img)
        
# print(i)

