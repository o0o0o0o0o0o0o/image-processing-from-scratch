#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def lines_detector_hough(edge,ThetaDim = None,DistStep = None,threshold = None,halfThetaWindowSize = 2,halfDistWindowSize = None):
    '''
    :param edge: 经过边缘检测得到的二值图
    :param ThetaDim: hough空间中theta轴的刻度数量(将[0,pi)均分为多少份),反应theta轴的粒度,越大粒度越细
    :param DistStep: hough空间中dist轴的划分粒度,即dist轴的最小单位长度
    :param threshold: 投票表决认定存在直线的起始阈值
    :return: 返回检测出的所有直线的参数(theta,dist)
    @author: bilibili-会飞的吴克
    '''
    imgsize = edge.shape
    if ThetaDim == None:
        ThetaDim = 90
    if DistStep == None:
        DistStep = 1
    MaxDist = np.sqrt(imgsize[0]**2 + imgsize[1]**2)
    DistDim = int(np.ceil(MaxDist/DistStep))

    if halfDistWindowSize == None:
        halfDistWindowSize = int(DistDim/50)
    accumulator = np.zeros((ThetaDim,DistDim)) # theta的范围是[0,pi). 在这里将[0,pi)进行了线性映射.类似的,也对Dist轴进行了线性映射

    sinTheta = [np.sin(t*np.pi/ThetaDim) for t in range(ThetaDim)]
    cosTheta = [np.cos(t*np.pi/ThetaDim) for t in range(ThetaDim)]

    for i in range(imgsize[0]):
        for j in range(imgsize[1]):
            if not edge[i,j] == 0:
                for k in range(ThetaDim):
                    accumulator[k][int(round((i*cosTheta[k]+j*sinTheta[k])*DistDim/MaxDist))] += 1

    M = accumulator.max()

    if threshold == None:
        threshold = int(M*2.3875/10)
    result = np.array(np.where(accumulator > threshold)) # 阈值化
    temp = [[],[]]
    for i in range(result.shape[1]):
        eight_neiborhood = accumulator[max(0, result[0,i] - halfThetaWindowSize + 1):min(result[0,i] + halfThetaWindowSize, accumulator.shape[0]), max(0, result[1,i] - halfDistWindowSize + 1):min(result[1,i] + halfDistWindowSize, accumulator.shape[1])]
        if (accumulator[result[0,i],result[1,i]] >= eight_neiborhood).all():
            temp[0].append(result[0,i])
            temp[1].append(result[1,i])

    result = np.array(temp)    # 非极大值抑制

    result = result.astype(np.float64)
    result[0] = result[0]*np.pi/ThetaDim
    result[1] = result[1]*MaxDist/DistDim

    return result

def drawLines(lines,edge,color = (255,0,0),err = 3):
    if len(edge.shape) == 2:
        result = np.dstack((edge,edge,edge))
    else:
        result = edge
    Cos = np.cos(lines[0])
    Sin = np.sin(lines[0])

    for i in range(edge.shape[0]):
        for j in range(edge.shape[1]):
            e = np.abs(lines[1] - i*Cos - j*Sin)
            if (e < err).any():
                result[i,j] = color

    return result


if __name__=='__main__':
    pic_path = './HoughImg/'
    pics = os.listdir(pic_path)

    for i in pics:
        if i[-5:] == '.jpeg' or i[-4:] == '.jpg':
            img = plt.imread(pic_path+i)
            blurred = cv2.GaussianBlur(img, (3, 3), 0)
            plt.imshow(blurred,cmap='gray')
            plt.axis('off')
            plt.show()

            if not len(blurred.shape) == 2:
                gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
            else:
                gray = blurred
            edge = cv2.Canny(gray, 50, 150)   #  二值图 (0 或 255) 得到 canny边缘检测的结果


            lines = lines_detector_hough(edge)
            final_img = drawLines(lines,blurred)
            

            plt.imshow(final_img,cmap='gray')
            plt.axis('off')
            plt.show()

