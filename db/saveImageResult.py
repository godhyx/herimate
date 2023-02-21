import cv2
import numpy as np
import os
from PIL import Image
from scipy import interpolate

import json
import math
import shutil

annotation_src = "/home/gy/Data/LD_VIT/order_ok/"
pic_src = "/home/gy/Data/LD_VIT/order_ok/"
store_pic = "/home/gy/Data/LD_VIT/order_ok_result/"

if not os.path.exists(store_pic):
    os.makedirs(store_pic)

h_samples= []
for i in range(210, 720, 10):
    h_samples.append(i)

RED = (0, 0, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (115, 181, 34)
BLUE = (255, 0, 0)
CYAN = (255, 128, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
PINK   = (180, 105, 255)
P10 = (255, 150, 100)
P11 = (100, 255, 100)
COLOR = [RED, GREEN, DARK_GREEN, BLUE, CYAN, YELLOW, ORANGE, PURPLE, PINK]
cate_list=['SingleSolid','SingleDotted','DoubleSolid','DoubleDotted','SolidDotted','DottedSolid','Roadline', 'Fence']
num=0
for files in os.listdir(annotation_src):
    if files[-4:] != 'json':
        continue
    print(files)
    # files = 'HK_20220228_001507_0072.mp4_2400.json'
    img = cv2.imread(pic_src+files[:-5]+'.jpg')
    img_h = img.shape[0]
    img_w = img.shape[1]
    # print(img_h, img_w)
    if not os.path.exists(pic_src+files[:-5]+'.jpg'):
        store_src='/home/gy/LD_VIT/MLDS_TrainingDataSet_960_error/'
        if not os.path.exists(store_src):
            os.makedirs(store_src)
        shutil.move(annotation_src+files, store_src+files)

    category = []
    cate_num = [0] * (len(cate_list)*10+len(cate_list))
    lanes = []
    y_samples = []
    breakpoints = []
    with open(annotation_src+files, 'r', encoding='utf-8') as anno_obj:
        data = json.load(anno_obj)
    # if 'shapes' in data:
    dataList = data['shapes']
    dataflag = [0]*len(dataList)
    for j in range(len(dataList)):
        if dataflag[j] == 1:
            continue
        if len(dataList[j]['points']) <2:
            continue
        labellist = dataList[j]['label'].split('-')
        if labellist[0] == 'DoubleSingleSolid' or labellist[0] == 'DoubleSingleDotted':
            continue
        if labellist[0] == 'ShortDotted':
            labellist[0] = 'SingleDotted'
        if labellist[0] == 'ForkSolid':   
            labellist[0] = 'SingleSolid'
        # properties = dataList[j]['properties']
        org_lanes = dataList[j]['points']
        label0 = label_u = labellist[0]
        label1 = label_l = labellist[0]
        breakpoint = org_lanes[-1]
        groupId = dataList[j]['group_id'] 
        if groupId:  #len(properties)>0:
            # groupId = dataList[j]['groupId']
            label_flag = 0
            for i in range(j+1, len(dataList)):
                if not dataList[i]['group_id'] :
                    continue
                labellist = dataList[i]['label'].split('-')
                if len(dataList[i]['points']) <2:
                    continue
                if labellist[0] == 'DoubleSingleSolid' or labellist[0] == 'DoubleSingleDotted':
                    continue
                if labellist[0] == 'ShortDotted':
                    labellist[0] = 'SingleDotted'
                if labellist[0] == 'ForkSolid': 
                    labellist[0] = 'SingleSolid'
                if dataList[i]['group_id']  == groupId:
                    if dataList[i]['points'][0][1]>org_lanes[0][1]:
                        if labellist[0] == label_u:
                            breakpoint = org_lanes[-1]
                            label_flag = 0
                        else:
                            breakpoint = dataList[i]['points'][-1]
                            label_flag = 1
                        label0 = labellist[0]  #dataList[i]['label'].split('-')[0]
                        label1 = label_u
                        org_lanes = dataList[i]['points'] + org_lanes
                        label_u = label0
                        label_l = label1
                    else:
                        if label_flag == 1 or label_flag == 1:
                            # if labellist[0] == label_l:
                            #     breakpoint = dataList[i]['points'][-1]
                            # else:
                            breakpoint = breakpoint
                            label0 = label_u
                            label1 = label_l
                        else:
                            if labellist[0] == label_l:
                                breakpoint = dataList[i]['points'][-1]
                                label_flag = 0
                            else:
                                breakpoint = org_lanes[-1]
                                label_flag = 1
                            label0 = label_u
                            label1 = labellist[0]  #dataList[i]['label'].split('-')[0]
                        org_lanes = org_lanes + dataList[i]['points']
                        label_u = label0
                        label_l = label1
                    dataflag[i] = 1
                    # label_flag = 1

        org_lane = []
        for i in range(len(org_lanes)-1, -1, -1):
            org_lane.append(org_lanes[i])

        if label0 == 'Roadline':
            label1 == 'Roadline'
            breakpoint = org_lane[0]
        if label0 == 'Fence':
            label1 == 'Fence'
            breakpoint = org_lane[0]

        c0 = cate_list.index(label0)
        c1 = cate_list.index(label1)
        category.append((c0+1)*10+(c1+1))
        cate_num[c0*10+c1] += 1

        lane = []
        y_smaple = []
        if breakpoint[1]<img_h:
            lane.append(breakpoint)
        for i in range(len(org_lane)-1):
            if org_lane[i][1]<img_h:
                lane.append(org_lane[i])
            # print(org_lane[i+1][1] - org_lane[i][1])
            dist = math.sqrt((org_lane[i+1][1] - org_lane[i][1])**2+(org_lane[i+1][0] - org_lane[i][0])**2)
            if dist >30:
                # print(org_lane[i])
                yp = [org_lane[i][1], org_lane[i+1][1]]
                xp = [org_lane[i][0], org_lane[i+1][0]]
                fun=interpolate.interp1d(yp,xp,kind="slinear")
                y = org_lane[i][1]
                inter_num = dist/20   #interpolate num
                detalY = (org_lane[i+1][1] - org_lane[i][1])/inter_num
                while 1:
                    y = y+detalY
                    if y < org_lane[i+1][1] and y<img_h:
                        x=fun(y)
                        lane.append([int(x),y])
                    else:
                        break
        if org_lane[-1][1]<img_h:
            lane.append(org_lane[-1])
        for i in range(len(lane)):
            y_smaple.append(lane[i][1])
        if len(lane)<3:
            continue
        lanes.append(lane)
        y_samples.append(y_smaple)


    for i, lane in enumerate(lanes):
        # print(lanes[i])
        cls = category[i]
        cls0 = cls//10
        cls1 = cls%10
        # print(cls0, cls1)
        breakpoints = lane[0]
        lane = lane[1:]
        lower, upper = lane[0][1], lane[-1][1]
        # print(lower, upper)
        assert lower<192 and upper<192
        
        for j in range(len(lane)-1): #-1
            # print(lanes[i][j])
            if cls0 == 0 or cls1 == 0:
                continue
            if lane[j][1]>=breakpoints[1]:
                color = COLOR[cls0-1]
            else:
                color = COLOR[cls1-1]
            cv2.line(img, (int(lane[j][0]), int(lane[j][1])), (int(lane[j+1][0]), int(lane[j+1][1])), color, 1) 
            # cv2.circle(img, (int(lane[j][0]), int(lane[j][1])), 1, color, 3)
        cv2.circle(img, (int(breakpoints[0]), int(breakpoints[1])), 2, (0,255,255), 5)  

    cv2.imwrite(store_pic+files[:-5]+'.jpg', img)
    
            

            