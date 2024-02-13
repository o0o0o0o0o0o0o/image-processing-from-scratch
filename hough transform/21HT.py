#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


def convolve(filter,mat,padding,strides):
    '''
    :param filter:卷积核，必须为二维(2 x 1也算二维) 否则返回None
    :param mat:图片
    :param padding:对齐
    :param strides:移动步长
    :return:返回卷积后的图片。(灰度图，彩图都适用)
    @author:bilibili-会飞的吴克
    '''
    result = None
    filter_size = filter.shape
    mat_size = mat.shape
    if len(filter_size) == 2:
        if len(mat_size) == 3:
            channel = []
            for i in range(mat_size[-1]):
                pad_mat = np.pad(mat[:,:,i], ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
                temp = []
                for j in range(0,mat_size[0],strides[1]):
                    temp.append([])
                    for k in range(0,mat_size[1],strides[0]):
                        val = (filter*pad_mat[j:j+filter_size[0],k:k+filter_size[1]]).sum()
                        temp[-1].append(val)
                channel.append(np.array(temp))

            channel = tuple(channel)
            result = np.dstack(channel)
        elif len(mat_size) == 2:
            channel = []
            pad_mat = np.pad(mat, ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
            for j in range(0, mat_size[0], strides[1]):
                channel.append([])
                for k in range(0, mat_size[1], strides[0]):
                    val = (filter * pad_mat[j:j + filter_size[0],k:k + filter_size[1]]).sum()
                    channel[-1].append(val)


            result = np.array(channel)


    return result


def AHTforCircles(edge,center_threhold_factor = None,score_threhold = None,min_center_dist = None,minRad = None,maxRad = None,center_axis_scale = None,radius_scale = None,halfWindow = None,max_circle_num = None):
    if center_threhold_factor == None:
        center_threhold_factor = 10.0
    if score_threhold == None:
        score_threhold = 15.0
    if min_center_dist == None:
        min_center_dist = 80.0
    if minRad == None:
        minRad = 0.0
    if maxRad == None:
        maxRad = 1e7*1.0
    if center_axis_scale == None:
        center_axis_scale = 1.0
    if radius_scale == None:
        radius_scale = 1.0
    if halfWindow == None:
        halfWindow = 2
    if max_circle_num == None:
        max_circle_num = 6
    min_center_dist_square = min_center_dist**2


    sobel_kernel_y = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    sobel_kernel_x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    edge_x = convolve(sobel_kernel_x,edge,[1,1,1,1],[1,1])
    edge_y = convolve(sobel_kernel_y,edge,[1,1,1,1],[1,1])


    center_accumulator = np.zeros((int(np.ceil(center_axis_scale*edge.shape[0])),int(np.ceil(center_axis_scale*edge.shape[1]))))
    k = np.array([[r for c in range(center_accumulator.shape[1])] for r in range(center_accumulator.shape[0])])
    l = np.array([[c for c in range(center_accumulator.shape[1])] for r in range(center_accumulator.shape[0])])
    minRad_square = minRad**2
    maxRad_square = maxRad**2
    points = [[],[]]

    edge_x_pad = np.pad(edge_x,((1,1),(1,1)),'constant')
    edge_y_pad = np.pad(edge_y,((1,1),(1,1)),'constant')
    Gaussian_filter_3 = 1.0 / 16 * np.array([(1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)])

    for i in range(edge.shape[0]):
        for j in range(edge.shape[1]):
            if not edge[i,j] == 0:
                dx_neibor = edge_x_pad[i:i+3,j:j+3]
                dy_neibor = edge_y_pad[i:i+3,j:j+3]
                dx = (dx_neibor*Gaussian_filter_3).sum()
                dy = (dy_neibor*Gaussian_filter_3).sum()
                if not (dx == 0 and dy == 0):
                    t1 = (k/center_axis_scale-i)
                    t2 = (l/center_axis_scale-j)
                    t3 = t1**2 + t2**2
                    temp = (t3 > minRad_square)&(t3 < maxRad_square)&(np.abs(dx*t1-dy*t2) < 1e-4)
                    center_accumulator[temp] += 1
                    points[0].append(i)
                    points[1].append(j)

    M = center_accumulator.mean()
    for i in range(center_accumulator.shape[0]):
        for j in range(center_accumulator.shape[1]):
            neibor = \
                center_accumulator[max(0, i - halfWindow + 1):min(i + halfWindow, center_accumulator.shape[0]),
                max(0, j - halfWindow + 1):min(j + halfWindow, center_accumulator.shape[1])]
            if not (center_accumulator[i,j] >= neibor).all():
                center_accumulator[i,j] = 0
                                                                        # 非极大值抑制

    plt.imshow(center_accumulator,cmap='gray')
    plt.axis('off')
    plt.show()

    center_threshold = M * center_threhold_factor
    possible_centers = np.array(np.where(center_accumulator > center_threshold))  # 阈值化


    sort_centers = []
    for i in range(possible_centers.shape[1]):
        sort_centers.append([])
        sort_centers[-1].append(possible_centers[0,i])
        sort_centers[-1].append(possible_centers[1, i])
        sort_centers[-1].append(center_accumulator[sort_centers[-1][0],sort_centers[-1][1]])

    sort_centers.sort(key=lambda x:x[2],reverse=True)

    centers = [[],[],[]]
    points = np.array(points)
    for i in range(len(sort_centers)):
        radius_accumulator = np.zeros(
            (int(np.ceil(radius_scale * min(maxRad, np.sqrt(edge.shape[0] ** 2 + edge.shape[1] ** 2)) + 1))),dtype=np.float32)
        if not len(centers[0]) < max_circle_num:
            break
        iscenter = True
        for j in range(len(centers[0])):
            d1 = sort_centers[i][0]/center_axis_scale - centers[0][j]
            d2 = sort_centers[i][1]/center_axis_scale - centers[1][j]
            if d1**2 + d2**2 < min_center_dist_square:
                iscenter = False
                break

        if not iscenter:
            continue

        temp = np.sqrt((points[0,:] - sort_centers[i][0] / center_axis_scale) ** 2 + (points[1,:] - sort_centers[i][1] / center_axis_scale) ** 2)
        temp2 = (temp > minRad) & (temp < maxRad)
        temp = (np.round(radius_scale * temp)).astype(np.int32)
        for j in range(temp.shape[0]):
            if temp2[j]:
                radius_accumulator[temp[j]] += 1
        for j in range(radius_accumulator.shape[0]):
            if j == 0 or j == 1:
                continue
            if not radius_accumulator[j] == 0:
                radius_accumulator[j] = radius_accumulator[j]*radius_scale/np.log(j) #radius_accumulator[j]*radius_scale/j
        score_i = radius_accumulator.argmax(axis=-1)
        if radius_accumulator[score_i] < score_threhold:
            iscenter = False

        if iscenter:
            centers[0].append(sort_centers[i][0]/center_axis_scale)
            centers[1].append(sort_centers[i][1]/center_axis_scale)
            centers[2].append(score_i/radius_scale)


    centers = np.array(centers)
    centers = centers.astype(np.float64)

    return centers








def drawCircles(circles,edge,color = (0,0,255),err = 600):
    if len(edge.shape) == 2:
        result = np.dstack((edge,edge,edge))
    else:
        result = edge
    for i in range(edge.shape[0]):
        for j in range(edge.shape[1]):
            dist_square = (circles[0]-i)**2 + (circles[1]-j)**2
            e = np.abs(circles[2]**2 - dist_square)
            if (e < err).any():
                result[i,j] = color
            if (dist_square < 25.0).any():
                result[i,j] = (255,0,0)
    return result


if __name__=='__main__':
    pic_path = './HoughCircleImg/'
    pics = os.listdir(pic_path)


    params = {
        '1.jpeg':{
        'center_threhold_factor': 3.33,
        'score_threhold':15.0,
         'min_center_dist':80.0,
         'minRad':0.0,
         'maxRad':1e7*1.0,
         'center_axis_scale':1.0,
         'radius_scale':1.0,
         'halfWindow':2,
        'max_circle_num':1
         },
        '4.jpg':{
        'center_threhold_factor': 2.0,
        'score_threhold': 15.0,
         'min_center_dist': 80.0,
         'minRad': 0.0,
         'maxRad': 1e7 * 1.0,
         'center_axis_scale': 1.0,
         'radius_scale': 1.0,
         'halfWindow': 2,
         'max_circle_num': 6
         },
        '2.jpeg':{
        'center_threhold_factor': 3.33,
        'score_threhold': 50.0,
         'min_center_dist': 80.0,
         'minRad': 0.0,
         'maxRad': 1e7 * 1.0,
         'center_axis_scale': 0.9,
         'radius_scale': 1.0,
         'halfWindow': 2,
         'max_circle_num': 1
         },
        '3.jpeg':{
        'center_threhold_factor': 1.5,
        'score_threhold': 56.0,
         'min_center_dist': 80.0,
         'minRad': 0.0,
         'maxRad': 1e7 * 1.0,
         'center_axis_scale': 0.8,
         'radius_scale': 1.0,
         'halfWindow': 2,
         'max_circle_num': 1
         },
        '0.jpg':{
        'center_threhold_factor': 1.5,
        'score_threhold': 30.0,
         'min_center_dist': 80.0,
         'minRad': 0.0,
         'maxRad': 1e7 * 1.0,
         'center_axis_scale': 1.0,
         'radius_scale': 1.0,
         'halfWindow': 2,
         'max_circle_num': 1
         }
    }
    for i in pics:
        if i[-5:] == '.jpeg' or i[-4:] == '.jpg':
            img = plt.imread(pic_path+i)
            blurred = cv2.GaussianBlur(img, (3, 3), 0)
            plt.imshow(blurred)
            plt.axis('off')
            plt.show()

            if not len(blurred.shape) == 2:
                gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
            else:
                gray = blurred
            edge = cv2.Canny(gray, 50, 150)   #  二值图 (0 或 255) 得到 canny边缘检测的结果

            circles = AHTforCircles(edge,
                                    center_threhold_factor=params[i]['center_threhold_factor'],score_threhold=params[i]['score_threhold'],min_center_dist=params[i]['min_center_dist'],minRad=params[i]['minRad'],
                                    maxRad=params[i]['maxRad'],center_axis_scale=params[i]['center_axis_scale'],radius_scale=params[i]['radius_scale'],
                                    halfWindow=params[i]['halfWindow'],max_circle_num=params[i]['max_circle_num'])
            final_img = drawCircles(circles,blurred)

            plt.imshow(final_img)
            plt.axis('off')
            plt.show()
