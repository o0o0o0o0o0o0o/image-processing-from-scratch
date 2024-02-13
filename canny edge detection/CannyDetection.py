#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import math
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

def linear_convolve(filter,mat,padding=None,strides=[1,1]):
    '''
    :param filter:线性卷积核
    :param mat:图片
    :param padding:对齐
    :param strides:移动步长
    :return:返回卷积后的图片。(灰度图，彩图都适用) 若不是线性卷积核，返回None
    @author:bilibili-会飞的吴克
    '''
    result = None
    filter_size = filter.shape
    if len(filter_size) == 2 and 1 in filter_size:
        if padding == None or len(padding) < 2:
            if filter_size[1] == 1:
                padding = [filter_size[0]//2,filter_size[0]//2]
            elif filter_size[0] == 1:
                padding = [filter_size[1]//2,filter_size[1]//2]
        if filter_size[0] == 1:
            result = convolve(filter,mat,[0,0,padding[0],padding[1]],strides)
        elif filter_size[1] == 1:
            result = convolve(filter, mat, [padding[0],padding[1],0,0], strides)

    return result

def _2_dim_divided_convolve(filter,mat):
    '''

    :param filter: 线性卷积核,必须为二维(2 x 1也算二维) 否则返回None
    :param mat: 图片
    :return: 卷积后的图片,(灰度图，彩图都适用) 若不是线性卷积核，返回None
    '''
    result = None
    if 1 in filter.shape:
        result = linear_convolve(filter,mat)
        result = linear_convolve(filter.T,result)

    return result

def judgeConnect(m2,threshold):
    e = 0.01
    s = []
    cood = []
    for i in range(m2.shape[0]):
        cood.append([])
        for j in range(m2.shape[1]):
            cood[-1].append([i,j])
            if abs(m2[i,j] - 255) < e:
                s.append([i,j])
    cood = np.array(cood)

    while not len(s) == 0:
        index = s.pop()
        jud = m2[max(0, index[0] - 1):min(index[0] + 2, m2.shape[1]), max(0, index[1] - 1):min(index[1] + 2, m2.shape[0])]
        jud_i = cood[max(0, index[0] - 1):min(index[0] + 2, cood.shape[1]), max(0, index[1] - 1):min(index[1] + 2, cood.shape[0])]
        jud = (jud > threshold[0])&(jud < threshold[1])
        jud_i = jud_i[jud]
        for i in range(jud_i.shape[0]):
            s.append(list(jud_i[i]))
            m2[jud_i[i][0],jud_i[i][1]] = 255

    return m2


def DecideAndConnectEdge(g_l,g_t,threshold = None):
    if threshold == None:
        lower_boundary = g_l.mean()*0.5
        threshold = [lower_boundary,lower_boundary*3]
    result = np.zeros(g_l.shape)

    for i in range(g_l.shape[0]):
        for j in range(g_l.shape[1]):
            isLocalExtreme = True
            eight_neiborhood = g_l[max(0,i-1):min(i+2,g_l.shape[0]),max(0,j-1):min(j+2,g_l.shape[1])]
            if eight_neiborhood.shape == (3,3):
                if g_t[i,j] <= -1:
                    x = 1/g_t[i,j]
                    first = eight_neiborhood[0,1] + (eight_neiborhood[0,1] - eight_neiborhood[0,0])*x
                    x = -x
                    second = eight_neiborhood[2,1] + (eight_neiborhood[2,2] - eight_neiborhood[2,1])*x
                    if not (g_l[i,j] > first and g_l[i,j] > second):
                        isLocalExtreme = False
                elif g_t[i,j] >= 1:
                    x = 1 / g_t[i, j]
                    first = eight_neiborhood[0, 1] + (eight_neiborhood[0, 2] - eight_neiborhood[0, 1]) * x
                    x = -x
                    second = eight_neiborhood[2, 1] + (eight_neiborhood[2, 1] - eight_neiborhood[2, 0]) * x
                    if not (g_l[i, j] > first and g_l[i, j] > second):
                        isLocalExtreme = False
                elif g_t[i,j] >= 0 and g_t[i,j] < 1:
                    y = g_t[i, j]
                    first = eight_neiborhood[1, 2] + (eight_neiborhood[0, 2] - eight_neiborhood[1, 2]) * y
                    y = -y
                    second = eight_neiborhood[1, 0] + (eight_neiborhood[1, 0] - eight_neiborhood[2, 0]) * y
                    if not (g_l[i, j] > first and g_l[i, j] > second):
                        isLocalExtreme = False
                elif g_t[i,j] < 0 and g_t[i,j] > -1:
                    y = g_t[i, j]
                    first = eight_neiborhood[1, 2] + (eight_neiborhood[1, 2] - eight_neiborhood[2, 2]) * y
                    y = -y
                    second = eight_neiborhood[1, 0] + (eight_neiborhood[0, 0] - eight_neiborhood[1, 0]) * y
                    if not (g_l[i, j] > first and g_l[i, j] > second):
                        isLocalExtreme = False
            if isLocalExtreme:
                result[i,j] = g_l[i,j]       #非极大值抑制

    result[result>=threshold[1]] = 255
    result[result<=threshold[0]] = 0


    result = judgeConnect(result,threshold)
    result[result!=255] = 0
    return result

def OneDimensionStandardNormalDistribution(x,sigma):
    E = -0.5/(sigma*sigma)
    return 1/(math.sqrt(2*math.pi)*sigma)*math.exp(x*x*E)

if __name__ == '__main__':

    # Gaussian_filter_3 = 1.0/16*np.array([(1,2,1),(2,4,2),(1,2,1)]) #Gaussian smoothing kernel when sigma = 0.8, size: 3x3

    # Gaussian_filter_5 = 1.0/159*np.array([
    #     [2,4,5,4,2],
    #     [4,9,12,9,4],
    #     [5,12,15,12,5],
    #     [4,9,12,9,4],
    #     [2,4,5,4,2]
    # ])  #Gaussian smoothing kernel when sigma = 1.4, size: 5x5


    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    pic_path = './img/'
    pics = os.listdir(pic_path)

    for i in pics:
        if i[-4:] == '.png'or i[-4:] == '.jpg' or i[-5:] == '.jpeg':
            filename = pic_path + i
            img = plt.imread(filename)
            if i[-4:] == '.png':
                img = img*255
            img = img.mean(axis=-1)  #this is a way to get a gray image.

            sigma = 1.52
            dim = int(np.round(6*sigma+1))
            if dim % 2 == 0:
                dim += 1
            linear_Gaussian_filter = [np.abs(t - (dim//2)) for t in range(dim)]

            linear_Gaussian_filter = np.array([[OneDimensionStandardNormalDistribution(t,sigma) for t in linear_Gaussian_filter]])
            linear_Gaussian_filter = linear_Gaussian_filter/linear_Gaussian_filter.sum()



            img2 = _2_dim_divided_convolve(linear_Gaussian_filter,img)
            # img2 = convolve(Gaussian_filter_5, img, [2, 2, 2, 2], [1, 1])

            plt.imshow(img2.astype(np.uint8), cmap='gray')
            plt.axis('off')
            plt.show()

            img3 = convolve(sobel_kernel_x,img2,[1,1,1,1],[1,1])
            img4 = convolve(sobel_kernel_y,img2,[1,1,1,1],[1,1])

            gradiant_length = (img3**2+img4**2)**(1.0/2)

            img3 = img3.astype(np.float64)
            img4 = img4.astype(np.float64)
            img3[img3==0]=0.00000001
            gradiant_tangent = img4/img3

            plt.imshow(gradiant_length.astype(np.uint8), cmap='gray')
            plt.axis('off')
            plt.show()

            #lower_boundary = 50
            final_img = DecideAndConnectEdge(gradiant_length,gradiant_tangent)

            cv2.imshow('edge',final_img.astype(np.uint8))
            cv2.waitKey(0)