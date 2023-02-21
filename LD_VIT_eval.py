"""
Evaluate the segmetation error of trained model on test set.
"""

import cv2
import os
import numpy as np
import math
import json
from scipy import interpolate
from sklearn.linear_model import LinearRegression


# from model import Model
# from att_model import AttModel
input_size_w = 960
input_size_h = 192
R = 0.5
MAX_DIS = 20
LABEL_DIR = '/home/vdb/SurroundViewData/04DataSet/TestDataSet/MLDS_TestSet/LD_MX_TestSet_v0.13.0/'
# LABEL_DIR = '/home/ps/Data/LD_VIT/test/'
PRED_POINTMAP_DIR = './results/test_result_float/'
# store_fp_fn_src = './results/fp_fn_src/'
# store_tp_src = './results/tp_src/'
store_file_path = './results/'
#CATE_LIST=['SingleSolid','SingleDotted','ForkSolid','DoubleSolid','DoubleDotted','SolidDotted','DottedSolid','Roadline', 'Fence']
CATE_LIST = ['Ignore','SingleSolid','SingleDotted','DoubleSolid','DoubleDotted','SolidDotted','DottedSolid','Roadline', 'Fence']
# CATE_LIST = ['Ignore','SingleSolid','SingleDotted','DoubleSolid','SolidDotted', 'DottedSolid', 'DoubleDotted', 'DotDashed','Roadline', 'Fence'] # 算法输出的顺序。目前没有分叉线类别，但有点虚
INFO_LIST = ['Weather', 'Ambient_light', 'Road_state', 'Vehicle_status', 'Interfere', 'Road_form']
# COLOR_LIST =  ['None', 'White', 'Yellow']
INFO_NUM = [4, 3, 3, 3, 8, 3]
# 0-Weather : 1晴天、2雨天、3雪天、4其他
# 1-Ambient_light： 1白天，2夜里，3逆光
# 2-Road_state（道路状态）：1高速-城快， 2乡村，3城市
# 3-Vehicle_status（本车状态）： 1车道内、2压线、3切角大（拐弯、掉头）
# 4-Interfere（干扰）：1无干扰、2道路标识干扰、3路口、4停车位干扰、5车道线磨损、6前车遮挡、7阴影、8杂线
# 5-Road_form：1直道、2坡道、3弯道

Saveflag_INFO = -1            # -1 无条件输出   有值只能输出一类
Saveflag_INFO_ITEM = -1      # 类型 是从1开始


def get_angle(xs, y_samples):
    lr = LinearRegression()
    xs, ys = xs[xs >= 0], y_samples[xs >= 0]
    if len(xs) > 1:
        lr.fit(ys[:, None], xs)
        k = lr.coef_[0]
        theta = np.arctan(k)
    else:
        theta = 0
    return theta

def line_accuracy(pred, gt, angle, angle_prev):

    if (abs (angle - angle_prev) > 1.0):
        return 200

    dist = 0
    num = 0
    invLen = 1.0 /(len(gt) - 1)
    dy = input_size_h * 1.0 / R / (len(gt) - 1)

    for i in range(len(gt)):
        if gt[i] < 0:
            continue
        if pred[i] < 0:
            continue
        num += 1

        if 1:           # 对于倾斜很大的边线，容易错误匹配，暂时不用该逻辑
            if i > 0 and gt[i - 1] > 0:
                x0 = gt[i - 1]
                y0 = -dy
            else:
                x0 = gt[i]
                y0 = 0

            if i + 1 < len(gt) and gt[i + 1] > 0:
                x1 = gt[i + 1]
                y1 = dy
            else:
                x1 = gt[i]
                y1 = 0

            if y1 - y0 > 1:
                ratio = (y1 - y0) / math.sqrt((y1 - y0) * (y1 - y0) + (x1 - x0) * (x1 - x0))
            else:
                ratio = abs(np.cos(angle))
        else:
            ratio = abs(np.cos(angle))

        ratio = abs(ratio * ((1 - i * invLen) * 0.9 + 0.4))
        dist += (abs(gt[i]-pred[i]) * ratio)


    if num>0:
        ave_dist = int(dist/num)
    else:
        ave_dist = 100
        # sum_num += np.where(np.abs(pred[i] - gt[i]) < thresh, 1., 0.)
        # k += 1
    return ave_dist  #sum_num/k


class DataEvaluator(object):

    def __init__(self):
        # h_samples= []
        h_samples = np.linspace(0, input_size_h/R, num=32)
        self.ys = h_samples

    
    def load_gt(self, label_path):
        """
        Create table of test samples and load groundtruth
        """
        h_samples = self.ys
        self.test_table = []
        self.label_dict = {}

        self. info_image_num = []
        for i in range(len(INFO_LIST)):
            self.info_image_num.append([0]*INFO_NUM[i])
        # print(self.info_image_num)
        # Add samples to the whole dataset
        for anno_file in os.listdir(label_path):
            if anno_file[-4:] != 'json':
                continue
            # print(anno_file)
            # anno_file = '2020-01-01-00-15-06-00_02_48-2022_03_01_15_45_15.json'
            category0 = []
            category1 = []
            breakpoints = []
            lanes = []
            angles = []
            samplename = anno_file[:-5]
            # samplename = '2020-01-01-00-03-19-00_01_18-2022_03_02_15_48_05'
            img_path = os.path.join(label_path, samplename+'.jpg')
            img   = cv2.imread(img_path)
            img_w = img.shape[1]
            img_h = img.shape[0]
            r = input_size_w / img_w
            with open(os.path.join(label_path, samplename+'.json'), 'r', encoding='utf-8') as anno_obj:
                data = json.load(anno_obj)
            if 'skyline' in data:
                skyline = data['skyline']
                crop_a = int(skyline-50/r)
            else:
                crop_a = int((img_h - input_size_h/r)*0.75)
            assert r == R
            img = img[crop_a:img_h,:]

            info = [0]*len(INFO_LIST)
            for i in range(len(INFO_LIST)):
                for j in range(len(INFO_LIST[i])):
                    if data['flags'][INFO_LIST[i]] == j+1:
                        self.info_image_num[i][j]  += 1
                        break
                info[i] = data['flags'][INFO_LIST[i]]
            # print(info)
            dataList = data['shapes']
            dataflag = [0]*len(dataList)
            for j in range(len(dataList)):
                xp=[]
                yp = []
                beginIdx = 0
                endIdx = 0
                if dataflag[j] == 1:
                    continue
                if len(dataList[j]['points']) <2:
                    continue
                labellist = dataList[j]['label'].split('-')
                if labellist[0] == 'DoubleSingleSolid' or labellist[0] == 'DoubleSingleDotted':
                    continue
                if labellist[0] == 'ShortDotted':
                    labellist[0] = 'SingleDotted'
                if labellist[0] == 'ForkSolid':   #分叉实线按实线算
                    labellist[0] = 'SingleSolid'
                org_lanes = dataList[j]['points']
                label0 = label_u = labellist[0]#dataList[j]['label'].split('-')[0]
                label1 = label_l = labellist[0]#dataList[j]['label'].split('-')[0]
                breakpoint = org_lanes[-1].copy()
                groupId = dataList[j]['group_id'] 
                if groupId:
                    label_flag = 0
                    for i in range(j+1, len(dataList)):
                        if not dataList[i]['group_id']:
                            continue
                        labellist = dataList[i]['label'].split('-')
                        if len(dataList[i]['points']) <2:
                            continue
                        if labellist[0] == 'DoubleSingleSolid' or labellist[0] == 'DoubleSingleDotted':
                            continue
                        if labellist[0] == 'ShortDotted':
                            labellist[0] = 'SingleDotted'   
                        if labellist[0] == 'ForkSolid':  #中间的分叉实线按实线算
                            labellist[0] = 'SingleSolid'
                        if dataList[i]['group_id'] == groupId:
                            if dataList[i]['points'][0][1]>org_lanes[0][1]:
                                if labellist[0] == label_u:
                                    breakpoint = org_lanes[-1].copy()
                                    label_flag = 0
                                else:
                                    breakpoint = dataList[i]['points'][-1].copy()
                                    label_flag = 1
                                label0 = labellist[0]#dataList[i]['label'].split('-')[0]
                                label1 = label_u
                                # breakpoint = dataList[i]['points'][-1]
                                org_lanes = dataList[i]['points'] + org_lanes
                                label_u = label0
                                label_l = label1
                            else:
                                if label_flag == 1 or label_flag == 1:
                                    # if labellist[0] == label_l:
                                    #     breakpoint = dataList[i]['points'][-1].copy()
                                    # else:
                                    #     breakpoint = breakpoint.copy()
                                    breakpoint = breakpoint.copy()
                                    label0 = label_u
                                    label1 = label_l
                                else:
                                    if labellist[0] == label_l:
                                        breakpoint = dataList[i]['points'][-1].copy()
                                        label_flag = 0
                                    else:
                                        breakpoint = org_lanes[-1].copy()
                                        label_flag = 1
                                    label0 = label_u
                                    label1 = labellist[0]#dataList[i]['label'].split('-')[0]
                                org_lanes = org_lanes + dataList[i]['points']
                                label_u = label0
                                label_l = label1
                            dataflag[i] = 1
                            # label_flag = 1
                if label0 == 'Roadline':
                    label1 = 'Roadline'
                    breakpoint = org_lanes[-1].copy()
                if label0 == 'Fence':
                    label1 = 'Fence'
                    breakpoint = org_lanes[-1].copy()

                if breakpoint == org_lanes[-1].copy():
                    breakpoint[1] = -2
                else:
                    breakpoint[1] -= crop_a
                breakpoints.append(breakpoint)


                org_lane = []
                for i in range(len(org_lanes)-1, -1, -1):
                    org_lane.append(org_lanes[i])

                c0 = CATE_LIST.index(label0)
                c1 = CATE_LIST.index(label1)
                category0.append(c0)  #c0+1
                category1.append(c1)  #c1+1

                for i in range(len(org_lane)):
                    xp.append(int(org_lane[i][0]))
                    yp.append(int(org_lane[i][1] - crop_a))
                if yp[0]<=h_samples[0] and yp[-1]<=h_samples[0]:
                    continue

                if (len(xp) <= 1):    #真值一个点：不参与比对
                    continue

                angle = get_angle(np.array(xp), np.array(yp))
                for i in range(len(h_samples)):
                    if h_samples[i] >= yp[0]:
                        beginIdx = i
                        break
                for i in range(len(h_samples)):
                    if h_samples[i] >= yp[-1]:
                        endIdx = i
                        break
                if endIdx == 0:
                    endIdx = len(h_samples)

                yvals = h_samples[beginIdx:endIdx]
                fun=interpolate.interp1d(yp,xp,kind="slinear")
                xinter=fun(yvals)

                xinterp = []
                for i in range(len(xinter)):
                    xinterp.append(int(xinter[i]))

                beginV = []
                endV = []
                if beginIdx > endIdx:
                    endIdx = beginIdx
                for i in range(len(h_samples)):
                    if i < beginIdx:
                        beginV.append(-2)
                    if i >= endIdx:
                        endV.append(-2)
                lane = beginV
                lane.extend(xinterp)
                lane.extend(endV)

                lanes.append(lane)
                angles.append(angle)
                
            self.label_dict[samplename] = {'image': img[:, :], 'gt_lanes': lanes, 'gt_breakpoint':breakpoints, 'gt_cls0': category0, 'gt_cls1': category1,
			                               'info':info, 'gt_angles':angles}
            self.test_table.append(samplename)
        # print(self.info_image_num)
        # print ('--Test set: ', len(self.test_table), ' samples.')

    def load_pred_txt(self, pred_path):
        """
        Create table of test samples and load groundtruth
        """
        self.test_pred = {}
        # Add samples to the whole dataset
        for samplename in self.test_table:
            label_f = os.path.join(pred_path, samplename+".txt")
            label = open(label_f, 'r')
            category0 = []
            category1 = []
            breakpoint = []
            lanes = []
            angles = []
            # Load the annotation information
            for line in label:
                lane=[]
                oneKps=[]
                line = line.strip()
                line = line.split()
                for i in range(5,len(line)):
                    lane.append(int(line[i]))
                cls0 = int(line[0])
                cls1 = int(line[1])
                bp_x = int(line[2])
                bp_y = int(line[3])
                lower = int(line[4])
                if bp_y-lower<6: #or cls0 == cls1:
                    bp_y = -2 
                # if cls0 == 
                breakpoint.append((bp_x, bp_y))
                category0.append(cls0)
                category1.append(cls1)
                assert cls0>0
                angle = get_angle(np.array(lane), np.array(self.ys))
                lanes.append(lane)
                angles.append(angle)
            self.test_pred[samplename] = {'pred_lanes': lanes, 'pred_breakpoint':breakpoint, 'pred_cls0': category0, 'pred_cls1': category1, 'pred_angles':angles}
    def gt_line(self, gt_lane, img, color):
        for k in range(len(gt_lane)-1):
            if gt_lane[k]<0:
                continue
            pt1 = (gt_lane[k], int(self.ys[k]))
            pt2 = (gt_lane[k+1], int(self.ys[k+1]))
            if gt_lane[k+1]<0:
                continue
            cv2.line(img, pt1, pt2, color, 2) 
    def pred_circle(self, pred_lane, img, color):
        for k in range(len(pred_lane)):
            if pred_lane[k]<0:
                continue
            pt1 = (pred_lane[k], int(self.ys[k]))
            cv2.circle(img, pt1, 3, color, -1)

    def acc_segm(self, dir):
        tp = fp = fn = 0
        bp_tp = bp_fp = bp_fn = 0
        info_tp = []
        info_fp = []
        info_fn = []
        info_cate = []
        info_cate_tp = []
        for i in range(len(INFO_LIST)):
            info_tp.append([0]*INFO_NUM[i])
            info_fp.append([0]*INFO_NUM[i])
            info_fn.append([0]*INFO_NUM[i])
            info_cate.append([0]*INFO_NUM[i])
            info_cate_tp.append([0]*INFO_NUM[i])
        # info_tp = [[0]*INFO_NUM]*len(INFO_LIST)
        # info_fp = [[0]*INFO_NUM]*len(INFO_LIST)
        # info_fn = [[0]*INFO_NUM]*len(INFO_LIST)
        # info_cate = [[0]*INFO_NUM]*len(INFO_LIST)
        tp_cate = [0]*len(CATE_LIST)
        cate_list = [0]*len(CATE_LIST)
        distance = []
        for i in range(MAX_DIS):
            distance.append(0)
        bp_distance = []
        for i in range(6):
            bp_distance.append(0)
        ys = self.ys

        for sample in evaluator.test_table:
            # sample = '2020-03-11-00-48-08.mp4_4800'
            img = self.label_dict[sample]['image']
            gt_lanes = self.label_dict[sample]['gt_lanes']
            gt_cls0 = self.label_dict[sample]['gt_cls0']
            gt_cls1 = self.label_dict[sample]['gt_cls1']
            gt_breakpoint = self.label_dict[sample]['gt_breakpoint']
            pred_lanes = self.test_pred[sample]['pred_lanes'] 
            pred_cls0 = self.test_pred[sample]['pred_cls0']
            pred_cls1 = self.test_pred[sample]['pred_cls1']
            pred_breakpoint = self.test_pred[sample]['pred_breakpoint']
            info = self.label_dict[sample]['info']
            angles = self.label_dict[sample]['gt_angles']
            angles_pred = self.test_pred[sample]['pred_angles']
            
            img1 = img.copy()
            tmpFN=0
            tmpFP=0
            bp_tmpFN=0
            bp_tmpFP=0

            # print(sample)
            tempMask = []
            for j in range(len(pred_lanes)):
                tempMask.append(0)
            gt_tempMask = []
            for i in range(len(gt_lanes)):
                gt_tempMask.append(0)
            #angles = [get_angle(np.array(x_gts), np.array(ys)) for x_gts in gt_lanes]
            # threshs = [MAX_DIS / np.cos(angle) for angle in angles]

            for i in range(len(gt_lanes)):
                if gt_cls0[i] == 0:
                    self.gt_line(gt_lanes[i], img1, (255,0,0))
                    continue
                if gt_tempMask[i] == 1:
                    continue
                minDist = 100
                maxindex = -1
                for j in range(len(pred_lanes)):
                    #if pred_cls[j] == 7:
                     #   continue
                    if tempMask[j] == 1:
                        continue
                    line_acc = line_accuracy(pred_lanes[j], gt_lanes[i], angles[i], angles_pred[j])

                    if gt_cls0[i] ==  CATE_LIST.index('Roadline') or gt_cls0[i] ==  CATE_LIST.index('Fence') :
                        if pred_cls0[j] ==  CATE_LIST.index('Roadline') or pred_cls0[j] ==  CATE_LIST.index('Fence') :
                            if line_acc < MAX_DIS:
                                line_acc = 0
                    if line_acc < minDist and line_acc < MAX_DIS:
                        minDist = line_acc
                        maxindex = j

                gt_minDist = minDist
                gt_maxindex = i  
                if maxindex > -1:
                    for k in range(len(gt_lanes)):
                        if gt_tempMask[k] == 1 or gt_cls0[k] == 0:
                            continue
                        line_acc = line_accuracy(pred_lanes[maxindex], gt_lanes[k], angles[k], angles_pred[maxindex])

                        if gt_cls0[k] == CATE_LIST.index('Roadline') or gt_cls0[k] == CATE_LIST.index('Fence'):
                            if pred_cls0[maxindex] == CATE_LIST.index('Roadline') or pred_cls0[maxindex] == CATE_LIST.index('Fence'):
                                if line_acc < MAX_DIS:
                                    line_acc = 0
                        # if line_acc<0:
                        #     break
                        if line_acc < gt_minDist:
                            gt_minDist = line_acc
                            gt_maxindex = k
                    if gt_maxindex == i:
                        distance = self.Kps_acc_segm(gt_lanes[gt_maxindex],pred_lanes[maxindex],distance, angles[gt_maxindex])
                        tp += 1
                        if gt_breakpoint[gt_maxindex][1]>0 and pred_breakpoint[maxindex][1]<0:
                            bp_fn += 1
                            bp_tmpFN = 1
                            cv2.circle(img1, (int(gt_breakpoint[gt_maxindex][0]),int(gt_breakpoint[gt_maxindex][1])), 15, (0,0,255), -1)
                        elif gt_breakpoint[gt_maxindex][1]<0 and pred_breakpoint[maxindex][1]>0:
                            bp_fp += 1
                            bp_tmpFP = 1
                            cv2.circle(img1, (int(pred_breakpoint[maxindex][0]), int(pred_breakpoint[maxindex][1])), 15, (0,0,255), 3)
                        elif gt_breakpoint[gt_maxindex][1]>0 and pred_breakpoint[maxindex][1]>0:
                            bp_tp += 1
                            cv2.circle(img1, (int(gt_breakpoint[gt_maxindex][0]),int(gt_breakpoint[gt_maxindex][1])), 15, (0,255,0), -1)
                            cv2.circle(img1, (int(pred_breakpoint[maxindex][0]), int(pred_breakpoint[maxindex][1])), 15, (0,255,0), 3)
                            dist = int(abs(math.sqrt((gt_breakpoint[gt_maxindex][0]-pred_breakpoint[maxindex][0])**2+(gt_breakpoint[gt_maxindex][1]-pred_breakpoint[maxindex][1])**2)))
                            for d in range(6):
                                if dist < (d+1)*5:
                                    bp_distance[d] += 1
                        tempMask[maxindex] = 1
                        gt_tempMask[gt_maxindex] = 1

                        for l in range(len(INFO_LIST)):
                            for n in range(INFO_NUM[l]):
                                if info[l] == n+1:
                                    info_tp[l][n] += 1
                                    info_cate[l][n] += 1
                                    if gt_cls0[gt_maxindex]  == pred_cls0[maxindex]:
                                        info_cate_tp[l][n] += 1
                        
                        for l in range(len(CATE_LIST)):
                            if gt_cls0[gt_maxindex] == l:
                                tp_cate[l] += 1
                                if pred_cls0[maxindex] == l:
                                    cate_list[l] +=1
                        self.gt_line(gt_lanes[gt_maxindex], img1, (0,255,0))
                        self.pred_circle(pred_lanes[maxindex], img1, (0,255,0))

            for i in range(len(gt_lanes)):
                if gt_tempMask[i] == 1 or gt_cls0[i] == 0:
                    continue
                minDist = 100
                maxindex = -1
                for j in range(len(pred_lanes)):
                    #if pred_cls[j] == 7:
                     #   continue
                    if tempMask[j] == 1:
                        continue
                    line_acc = line_accuracy(pred_lanes[j], gt_lanes[i], angles[i], angles_pred[j])

                    if gt_cls0[i] == CATE_LIST.index('Roadline') or gt_cls0[i] == CATE_LIST.index('Fence'):
                        if pred_cls0[j] == CATE_LIST.index('Roadline') or pred_cls0[j] == CATE_LIST.index('Fence'):
                            if line_acc < MAX_DIS:
                                line_acc = 0
                    # if line_acc<0:
                    #     break
                    if line_acc < minDist and line_acc<MAX_DIS:
                        minDist = line_acc
                        maxindex = j

                gt_minDist = minDist
                gt_maxindex = i  
                if maxindex > -1:
                    tp += 1
                    if gt_breakpoint[gt_maxindex][1]>0 and pred_breakpoint[maxindex][1]<0:
                        bp_fn += 1
                        bp_tmpFN = 1
                        cv2.circle(img1, (int(gt_breakpoint[gt_maxindex][0]), int(gt_breakpoint[gt_maxindex][1])), 15, (0,0,255), -1)
                    elif gt_breakpoint[gt_maxindex][1]<0 and pred_breakpoint[maxindex][1]>0:
                        bp_fp += 1
                        bp_tmpFN = 1
                        cv2.circle(img1, (int(pred_breakpoint[maxindex][0]), int(pred_breakpoint[maxindex][1])), 15, (0,0,255), 3)
                    elif gt_breakpoint[gt_maxindex][1]>0 and pred_breakpoint[maxindex][1]>0:
                        bp_tp += 1
                        cv2.circle(img1, (int(gt_breakpoint[gt_maxindex][0]), int(gt_breakpoint[gt_maxindex][1])), 15, (0,255,0), -1)
                        cv2.circle(img1, (int(pred_breakpoint[maxindex][0]), int(pred_breakpoint[maxindex][1])), 15, (0,255,0), 3)
                        dist = int(abs(math.sqrt((gt_breakpoint[gt_maxindex][0]-pred_breakpoint[maxindex][0])**2+(gt_breakpoint[gt_maxindex][1]-pred_breakpoint[maxindex][1])**2)))
                        for d in range(6):
                            if dist<(d+1)*5:
                                bp_distance[d] += 1
                        
                    tempMask[maxindex] = 1
                    gt_tempMask[gt_maxindex] = 1
                    for l in range(len(INFO_LIST)):
                        for n in range(INFO_NUM[l]):
                            if info[l] == n+1:
                                info_tp[l][n] += 1
                                info_cate[l][n] += 1
                                if gt_cls0[gt_maxindex]  == pred_cls0[maxindex]:
                                    info_cate_tp[l][n] += 1

                    for l in range(len(CATE_LIST)):
                        if gt_cls0[gt_maxindex] == l:
                            tp_cate[l] += 1
                            if pred_cls0[maxindex] == l:
                                cate_list[l] +=1
                    self.gt_line(gt_lanes[gt_maxindex], img1, (0,255,0))
                    self.pred_circle(pred_lanes[maxindex], img1, (0,255,0))
                else:
                    maxIndex = -1
                    if gt_cls0[i] != CATE_LIST.index('Roadline') and gt_cls0[i] != CATE_LIST.index('Fence'):
                        for k in range(len(gt_lanes)):
                            if i == k:
                                continue
                            if gt_cls0[k] != CATE_LIST.index('Roadline') and gt_cls0[k] != CATE_LIST.index('Fence'):
                                continue
                            line_acc = line_accuracy(gt_lanes[i], gt_lanes[k], angles[k], angles[i])
                            if line_acc<MAX_DIS:
                                maxIndex = k
                    else:  #是路沿围栏
                        for k in range(len(gt_lanes)):
                            if i == k:
                                continue
                            if gt_tempMask[k] == 0:    #路沿or围栏 存在匹配成功的相邻线
                                continue
                            line_acc = line_accuracy(gt_lanes[i], gt_lanes[k], angles[k], angles[i])
                            if line_acc < MAX_DIS:
                                maxIndex = k

                    if maxIndex == -1:
                        fn+=1
                        for l in range(len(INFO_LIST)):
                            for n in range(INFO_NUM[l]):
                                if info[l] == n+1:
                                    info_fn[l][n] += 1
                        tmpFN = 1
                        self.gt_line(gt_lanes[i], img1, (0,0,255))
                    else:
                        self.gt_line(gt_lanes[i], img1, (255,0,0))

            for j in range(len(pred_lanes)): 
                # if pred_cls[j] == 7:
                #     continue
                if tempMask[j]==1: # and cv2.contourArea(pred_contours[j])<30:
                    continue
                maxIndex = -1
                # if pred_cls0[j] != CATE_LIST.index('Roadline') and pred_cls0[j] != CATE_LIST.index('Fence'):
                for k in range(len(gt_lanes)):
                    if gt_cls0[k] != CATE_LIST.index('Roadline') and gt_cls0[k] != CATE_LIST.index('Fence') and gt_cls0[k] != 0:
                        continue
                    line_acc = line_accuracy(pred_lanes[j], gt_lanes[k], angles[k], angles_pred[j])
                    if line_acc<MAX_DIS:
                        maxIndex = k
                if maxIndex == -1:
                    fp += 1
                    for l in range(len(INFO_LIST)):
                        for n in range(INFO_NUM[l]):
                            if info[l] == n+1:
                                info_fp[l][n] += 1
                    tmpFP = 1
                    self.pred_circle(pred_lanes[j], img1, (0,0,255))
                else:
                    self.pred_circle(pred_lanes[j], img1, (255,0,0))

            if Saveflag_INFO >= 0 and Saveflag_INFO_ITEM > 0 and Saveflag_INFO_ITEM != info[Saveflag_INFO]:
                continue
            if not os.path.exists(store_file_path+'fp_fn_src/'+dir):
                os.makedirs(store_file_path+'fp_fn_src/'+dir)   
            if tmpFN == 1 or tmpFP ==1:
                cv2.imwrite(store_file_path+'fp_fn_src/'+dir+'/' + sample + '.jpg', img1)
            else:
                cv2.imwrite(store_file_path+'tp_src/'+dir +'/'+ sample + '.jpg', img1)

            if not os.path.exists(store_file_path+'breakpoint_fp_fn_src/'+dir):
                os.makedirs(store_file_path+'breakpoint_fp_fn_src/'+dir)
            if bp_tmpFN == 1 or bp_tmpFP == 1:
                img = cv2.resize(img, (input_size_w, input_size_h))
                img1 = cv2.resize(img1, (input_size_w, input_size_h))
                result = np.vstack((img, img1))
                cv2.imwrite(store_file_path+'breakpoint_fp_fn_src/'+dir+'/' + sample + '.jpg', result)

        return tp, fp, fn, bp_tp, bp_fp, bp_fn, cate_list, tp_cate, info_tp, info_fp, info_fn, info_cate, info_cate_tp, distance, bp_distance

    def Kps_acc_segm(self,gt_lanes,pred_lanes,distance, angles):  #,tp,tp_class,fn,fn_class):
        temp_fn=0
        tempDist = []
        for i in range(MAX_DIS):
            tempDist.append(0)
        ave_dist = line_accuracy(pred_lanes, gt_lanes, angles, angles)
        if ave_dist < MAX_DIS:
            for k in range(MAX_DIS):
                if ave_dist == k:
                    tempDist[k] += 1
        else:
            temp_fn = 1
        if temp_fn == 0:
            for k in range(MAX_DIS):
                distance[k] += tempDist[k]
        return distance

    def mAP_segm(self, f, dir, interpolate=True):
        tp, fp, fn, bp_tp, bp_fp, bp_fn, cate_list,tp_cate, info_tp, info_fp, info_fn, info_cate, info_cate_tp, distance, bp_distance = self.acc_segm(dir) 
        # print(tp, fp, fn)
        image_num = len(self.test_table)
        f.write('  TP FP FN All Accuracy:\n')
        f.write("ALL_line_Accuracy: ")
        if tp+fp+fn == 0:
            if image_num == 0:
                f.write(str(tp)+' '+str(fp)+' '+str(fn)+' '+str(tp+fp+fn)+' '+str(0)+' '+str(0)+'\n\n')
            else:
                f.write(str(tp)+' '+str(fp)+' '+str(fn)+' '+str(tp+fp+fn)+' '+str(0)+' '+str(round(fp/image_num*100, 2))+'%'+'\n\n')
        else:
            if image_num == 0:
                f.write(str(tp)+' '+str(fp)+' '+str(fn)+' '+str(tp+fp+fn)
                    +' '+str(round(tp/(tp+fp+fn)*100, 2))+'%'+' '+str(0)+'%\n\n')
            else:
                f.write(str(tp)+' '+str(fp)+' '+str(fn)+' '+str(tp+fp+fn)
                        +' '+str(round(tp/(tp+fp+fn)*100, 2))+'%'+' '+str(round(fp/image_num*100, 2))+'%\n\n')

        total_cate_list = 0
        total_tp_cate = 0
        for i in range(len(cate_list)):
            total_cate_list += cate_list[i]
            total_tp_cate += tp_cate[i]

        f.write("ALL_Category_Accuracy: ")
        if total_tp_cate == 0:
            f.write(str(total_cate_list)+' '+str(0)+' '+str(0)+' '+str(total_tp_cate)+' '+str(0)+'\n')
        else:
            f.write(str(total_cate_list)+' '+str(0)+' '+str(0)+' '+str(total_tp_cate)+' '+str(round(total_cate_list/total_tp_cate*100, 2))+'%\n')

        for i in range(len(cate_list)):
            f.write(CATE_LIST[i]+'_Category_Accuracy: ')
            if tp_cate[i] == 0:
                f.write(str(cate_list[i])+' '+str(0)+' '+str(0)+' '+str(tp_cate[i])+' '+str(0)+'\n')
            else:
                f.write(str(cate_list[i])+' '+str(0)+' '+str(0)+' '+str(tp_cate[i])+' '+str(round(cate_list[i]/tp_cate[i]*100, 2))+'%\n')
        f.write('\n')

        for i in range(len(INFO_LIST)):
            for j in range(INFO_NUM[i]):
                f.write(INFO_LIST[i]+'_'+str(j+1)+'_line_Accuracy: ')
                if info_tp[i][j]+info_fp[i][j]+info_fn[i][j]== 0:
                    if self.info_image_num[i][j] == 0:
                        f.write(str(info_tp[i][j])+' '+str(info_fp[i][j])+' '+str(info_fn[i][j])+' '+str(info_tp[i][j]+info_fp[i][j]+info_fn[i][j])
                                +' '+str(0)+' '+str(0)+'\n')
                    else:
                        f.write(str(info_tp[i][j])+' '+str(info_fp[i][j])+' '+str(info_fn[i][j])+' '+str(info_tp[i][j]+info_fp[i][j]+info_fn[i][j])
                                +' '+str(0)+' '+str(round(info_fp[i][j]/self.info_image_num[i][j]*100,2))+'%'+'\n')
                else:
                    if self.info_image_num[i][j] == 0:
                        f.write(str(info_tp[i][j])+' '+str(info_fp[i][j])+' '+str(info_fn[i][j])+' '+str(info_tp[i][j]+info_fp[i][j]+info_fn[i][j])
                            +' '+str(round(info_tp[i][j]/(info_tp[i][j]+info_fp[i][j]+info_fn[i][j])*100,2))+'%'
                            +' '+str(0)+'%\n')
                    else:
                        f.write(str(info_tp[i][j])+' '+str(info_fp[i][j])+' '+str(info_fn[i][j])+' '+str(info_tp[i][j]+info_fp[i][j]+info_fn[i][j])
                                +' '+str(round(info_tp[i][j]/(info_tp[i][j]+info_fp[i][j]+info_fn[i][j])*100,2))+'%'
                                +' '+str(round(info_fp[i][j]/self.info_image_num[i][j]*100,2))+'%\n')
                f.write(INFO_LIST[i]+'_'+str(j+1)+'_Category_Accuracy: ')
                if info_cate[i][j] == 0:
                    f.write(str(info_cate_tp[i][j])+' '+str(0)+' '+str(0)+' '+str(info_cate[i][j])+' '+str(0)+'\n\n')
                else:
                    f.write(str(info_cate_tp[i][j])+' '+str(0)+' '+str(0)+' '+str(info_cate[i][j])+' '+str(round(info_cate_tp[i][j]/info_cate[i][j]*100,2))+'%\n\n')
        percentage = []
        totDisVec = []
        totalDis = 0
        dis = 0
        for k in range(MAX_DIS):
            percentage.append(0)
        for k in range(MAX_DIS):
            totalDis += distance[k]
            totDisVec.append(totalDis)
        for k in range(MAX_DIS):
            dis +=  distance[k]
            percentage[k] = round((dis/tp)*100,2)
        f.write("point location accuracy statistics: \n")
        for k in range(MAX_DIS):
            f.write("distance "+str(k)+": "+str(totDisVec[k])+" "+ str(percentage[k]) +"%"+"\n")
        f.write('\n')

        f.write("BreakPoint_Accuracy: ")
        if bp_tp+bp_fp+bp_fn == 0:
            f.write(str(bp_tp)+' '+str(bp_fp)+' '+str(bp_fn)+' '+str(bp_tp+bp_fp+bp_fn)+' '+str(0)+'\n\n')
        else:
            f.write(str(bp_tp)+' '+str(bp_fp)+' '+str(bp_fn)+' '+str(bp_tp+bp_fp+bp_fn)+' '+str(round(bp_tp/(bp_tp+bp_fp+bp_fn)*100, 2))+'%\n\n')
        if bp_tp == 0:
            for i in range(6):
                f.write("distance "+str((i+1)*5)+": "+str(0)+'\n')
        else:
            for i in range(6):
                f.write("distance "+str((i+1)*5)+": "+str(round(bp_distance[i]/bp_tp*100, 2))+'%\n')
        return tp, fp, fn, bp_tp, bp_fp, bp_fn, cate_list,tp_cate, info_tp, info_fp, info_fn, info_cate, info_cate_tp, distance, bp_distance
                   
# Construct the evaluator
evaluator = DataEvaluator()
total_tp = total_fp = total_fn = 0
total_bp_tp = total_bp_fp = total_bp_fn = 0
total_cate_list = [0]*len(CATE_LIST)
total_tp_cate = [0]*len(CATE_LIST)
total_distance = [0]*MAX_DIS
total_bp_distance = [0]*6
total_info_tp = []
total_info_fp = []
total_info_fn = []
total_info_cate = []
total_info_cate_tp = []
total_info_image_num = []
total_image_num = 0
for i in range(len(INFO_LIST)):
    total_info_tp.append([0]*INFO_NUM[i])
    total_info_fp.append([0]*INFO_NUM[i])
    total_info_fn.append([0]*INFO_NUM[i])
    total_info_cate.append([0]*INFO_NUM[i])
    total_info_cate_tp.append([0]*INFO_NUM[i])
    total_info_image_num.append([0]*INFO_NUM[i])
            
# for i in range(MAX_DIS):
#     distance.append(0)
for maindir, subdir, file_name_list in os.walk(PRED_POINTMAP_DIR):
    if not subdir:
        dir = maindir.split('/')[-1]
        # dir = 'C'
        print(dir)
        evaluator.load_gt(LABEL_DIR+dir)
        print ("load_gt success")
        evaluator.load_pred_txt(PRED_POINTMAP_DIR+dir)
        print ("load_pred_txt success")
        f = open(store_file_path+dir+'_static_FN_FP.txt','w')
        tp, fp, fn, bp_tp, bp_fp, bp_fn, cate_list,tp_cate, info_tp, info_fp, info_fn, info_cate, info_cate_tp, distance, bp_distance = evaluator.mAP_segm(f, dir)
        f.close()
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_bp_tp += bp_tp
        total_bp_fp += bp_fp
        total_bp_fn += bp_fn
        for i in range(len(CATE_LIST)):
            total_cate_list[i] += cate_list[i]
            total_tp_cate[i] += tp_cate[i]
        total_image_num += len(evaluator.test_table)
        for i in range(len(INFO_LIST)):
            for j in range(INFO_NUM[i]):
                total_info_tp[i][j] += info_tp[i][j]
                total_info_fp[i][j] += info_fp[i][j]
                total_info_fn[i][j] += info_fn[i][j]
                total_info_cate[i][j] += info_cate[i][j]
                total_info_cate_tp[i][j] += info_cate_tp[i][j]
                total_info_image_num[i][j] += evaluator.info_image_num[i][j]
        # print(total_tp_cate)
        for i in range(MAX_DIS):
            total_distance[i] +=distance[i]
        for i in range(6):
            total_bp_distance[i] +=bp_distance[i]

f = open(store_file_path+'total_static_FN_FP.txt','w')
f.write('  TP FP FN All Accuracy:\n')
f.write("ALL_line_Accuracy: ")
if total_tp+total_fp+total_fn == 0:
    if total_image_num == 0:
        f.write(str(total_tp)+' '+str(total_fp)+' '+str(total_fn)+' '+str(total_tp+total_fp+total_fn)
        +' '+str(0)+' '+str(0)+'\n\n')
    else:
        f.write(str(total_tp)+' '+str(total_fp)+' '+str(total_fn)+' '+str(total_tp+total_fp+total_fn)
        +' '+str(0)+' '+str(round(total_fp/total_image_num*100, 2))+'%'+'\n\n')
else:
    if total_image_num == 0:
        f.write(str(total_tp)+' '+str(total_fp)+' '+str(total_fn)+' '+str(total_tp+total_fp+total_fn)
                +' '+str(round(total_tp/(total_tp+total_fp+total_fn)*100, 2))+'%'+' '+str(0)+'%\n\n')
    else:
        f.write(str(total_tp)+' '+str(total_fp)+' '+str(total_fn)+' '+str(total_tp+total_fp+total_fn)
                +' '+str(round(total_tp/(total_tp+total_fp+total_fn)*100, 2))+'%'+' '+str(round(total_fp/total_image_num*100, 2))+'%\n\n')


all_cate_list = 0
all_tp_cate = 0
for i in range(len(CATE_LIST)):
    all_cate_list += total_cate_list[i]
    all_tp_cate += total_tp_cate[i]

f.write("ALL_Category_Accuracy: ")
if total_tp_cate == 0:
    f.write(str(all_cate_list)+' '+str(0)+' '+str(0)+' '+str(all_tp_cate)+' '+str(0)+'\n')
else:
    f.write(str(all_cate_list)+' '+str(0)+' '+str(0)+' '+str(all_tp_cate)+' '+str(round(all_cate_list/all_tp_cate*100, 2))+'%\n')

for i in range(len(total_cate_list)):
    f.write(CATE_LIST[i]+'_Category_Accuracy: ')
    if total_tp_cate[i] == 0:
        f.write(str(total_cate_list[i])+' '+str(0)+' '+str(0)+' '+str(total_tp_cate[i])+' '+str(0)+'\n')
    else:
        f.write(str(total_cate_list[i])+' '+str(0)+' '+str(0)+' '+str(total_tp_cate[i])+' '+str(round(total_cate_list[i]/total_tp_cate[i]*100, 2))+'%\n')
f.write('\n')
for i in range(len(INFO_LIST)):
    for j in range(INFO_NUM[i]):
        f.write(INFO_LIST[i]+'_'+str(j+1)+'_line_Accuracy: ')
        if total_info_tp[i][j]+total_info_fp[i][j]+total_info_fn[i][j]== 0:
            if total_info_image_num[i][j] == 0:
                f.write(str(total_info_tp[i][j])+' '+str(total_info_fp[i][j])+' '+str(total_info_fn[i][j])+' '+str(total_info_tp[i][j]+total_info_fp[i][j]+total_info_fn[i][j])
                        +' '+str(0)+' '+str(0)+'\n')
            else:
                f.write(str(total_info_tp[i][j])+' '+str(total_info_fp[i][j])+' '+str(total_info_fn[i][j])+' '+str(total_info_tp[i][j]+total_info_fp[i][j]+total_info_fn[i][j])
                        +' '+str(0)+' '+str(round(total_info_fp[i][j]/total_info_image_num[i][j]*100,2))+'%'+'\n')
        else:
            if total_info_image_num[i][j] == 0:
                f.write(str(total_info_tp[i][j])+' '+str(total_info_fp[i][j])+' '+str(total_info_fn[i][j])+' '+str(total_info_tp[i][j]+total_info_fp[i][j]+total_info_fn[i][j])
                        +' '+str(round(total_info_tp[i][j]/(total_info_tp[i][j]+total_info_fp[i][j]+total_info_fn[i][j])*100,2))+'%'
                        +' '+str(0)+'%\n')
            else:
                f.write(str(total_info_tp[i][j])+' '+str(total_info_fp[i][j])+' '+str(total_info_fn[i][j])+' '+str(total_info_tp[i][j]+total_info_fp[i][j]+total_info_fn[i][j])
                +' '+str(round(total_info_tp[i][j]/(total_info_tp[i][j]+total_info_fp[i][j]+total_info_fn[i][j])*100,2))+'%'
                +' '+str(round(total_info_fp[i][j]/total_info_image_num[i][j]*100,2))+'%\n')
        f.write(INFO_LIST[i]+'_'+str(j+1)+'_Category_Accuracy: ')
        if total_info_cate[i][j] == 0:
            f.write(str(total_info_cate_tp[i][j])+' '+str(0)+' '+str(0)+' '+str(total_info_cate[i][j])+' '+str(0)+'\n\n')
        else:
            f.write(str(total_info_cate_tp[i][j])+' '+str(0)+' '+str(0)+' '+str(total_info_cate[i][j])+' '+str(round(total_info_cate_tp[i][j]/total_info_cate[i][j]*100,2))+'%\n\n')

percentage = []
totDisVec = []
totalDis = 0
dis = 0
for k in range(MAX_DIS):
    percentage.append(0)
for k in range(MAX_DIS):
    totalDis += total_distance[k]
    totDisVec.append(totalDis)
for k in range(MAX_DIS):
    dis +=  total_distance[k]
    percentage[k] = round((dis/total_tp)*100,2)
f.write("point location accuracy statistics: \n")
for k in range(MAX_DIS):
    f.write("distance "+str(k)+": "+str(totDisVec[k])+" "+ str(percentage[k]) +"%"+"\n")
f.write('\n')

f.write("BreakPoint_Accuracy: ")
if total_bp_tp+total_bp_fp+total_bp_fn == 0:
    f.write(str(total_bp_tp)+' '+str(total_bp_fp)+' '+str(total_bp_fn)+' '+str(total_bp_tp+total_bp_fp+total_bp_fn)+' '+str(0)+'\n\n')
else:
    f.write(str(total_bp_tp)+' '+str(total_bp_fp)+' '+str(total_bp_fn)+' '+str(total_bp_tp+total_bp_fp+total_bp_fn)+' '+str(round(total_bp_tp/(total_bp_tp+total_bp_fp+total_bp_fn)*100, 2))+'%\n\n')
if total_bp_tp == 0:
    for i in range(6):
        f.write("distance "+str((i+1)*5)+": "+str(0)+'\n')
else:
    for i in range(6):
        f.write("distance "+str((i+1)*5)+": "+str(round(total_bp_distance[i]/total_bp_tp*100, 2))+'%\n')
f.close()

