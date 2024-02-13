#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import math

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


def score_for_each_pixel(sq_img_gx,sq_img_gy,img_gx_gy,k):
    '''
    所传入的参数都只能有一个通道，且形状必须相同
    :param sq_img_gx: x方向上的梯度平方的图片
    :param sq_img_gy: y方向上的梯度平方的图片
    :param img_gx_gy: x,y方向上梯度乘积的图片
    :param k: 矩阵的迹前面的系数
    :return: 各点的得分
    '''
    result = []
    for i in range(sq_img_gx.shape[0]):
        result.append([])
        for j in range(sq_img_gx.shape[1]):
            M = np.array(
                [
                    [sq_img_gx[i,j],img_gx_gy[i,j]],
                    [img_gx_gy[i,j],sq_img_gy[i,j]]
                ]
            )
            result[-1].append(np.linalg.det(M)-k*(np.trace(M)**2))
    return np.array(result)

def Sign(img,score,area,decide_value=None,boder=[3,3,3,3]):
    '''
    :param img: 需要在角点处做标记的图片(可为多通道)
    :param score: 各个像素的角点得分
    :param area: 标记区域的大小(area[0] x area[1])
    :param decide_value: 决策是否为角点的阈值
    :param boder: 标记的边界宽度
    :return: 返回标记后的图片
    '''
    if decide_value == None:
        decide_value =  34*math.fabs(score.mean())  # 34这个参数可调
        print(decide_value)
    judger = score > decide_value
    final_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            isLocalExtreme = score[i,j] >= score[max(i-(area[0]//2),0):min(i+(area[0]//2)+1,img.shape[0]),max(j-(area[1]//2),0):min(j+(area[1]//2)+1,img.shape[1])] #非极值抑制
            if judger[i,j] and isLocalExtreme.all():
                for k in range(min(boder[0],area[1]//6+1)):
                    final_img[max(i-(area[0]//2),0):min(i+(area[0]//2)+1,img.shape[0]),max(j-(area[1]//2),0)+k,:] = [255,0,0] #top
                for k in range(min(boder[1],area[1]//6+1)):
                    final_img[max(i-(area[0]//2),0):min(i+(area[0]//2)+1,img.shape[0]),min(j+(area[1]//2),img.shape[1]-1)-k,:] = [255,0,0] #bottom
                for k in range(min(boder[2],area[0]//6+1)):
                    final_img[max(i-(area[0]//2),0)+k,max(j-(area[1]//2),0):min(j+(area[1]//2)+1,img.shape[1]),:] = [255, 0, 0] #left
                for k in range(min(boder[3],area[0]//6+1)):
                    final_img[min(i+(area[0]//2),img.shape[0]-1)-k,max(j-(area[1]//2),0):min(j+(area[1]//2)+1,img.shape[1]),:] = [255,0,0]  # right
    return final_img

def OneDimensionStandardNormalDistribution(x,sigma):
    E = -0.5/(sigma*sigma)
    return 1/(math.sqrt(2*math.pi)*sigma)*math.exp(x*x*E)

if __name__ == '__main__':

    pic_path = './img/'
    pics = os.listdir(pic_path)

    window = 1.0/159*np.array([
        [2,4,5,4,2],
        [4,9,12,9,4],
        [5,12,15,12,5],
        [4,9,12,9,4],
        [2,4,5,4,2]
    ])   # window(5x5 Gaussisan kernel)
    linear_Gaussian_filter_5 = [2, 1, 0, 1, 2]
    sigma = 1.4
    linear_Gaussian_filter_5 = np.array([[OneDimensionStandardNormalDistribution(t, sigma) for t in linear_Gaussian_filter_5]])
    linear_Gaussian_filter_5 = linear_Gaussian_filter_5/linear_Gaussian_filter_5.sum()


    G_y = np.array(
        [
            [2, 2, 4, 2, 2],
            [1, 1, 2, 1, 1],
            [0 ,0 ,0 ,0 ,0],
            [-1,-1,-2,-1,-1],
            [-2,-2,-4,-2,-2]
        ]
    )
    G_x = np.array(
        [
            [2, 1, 0, -1, -2],
            [2, 1, 0, -1, -2],
            [4, 2, 0, -2, -4],
            [2, 1, 0, -1, -2],
            [2, 1, 0, -1, -2]
        ]
    ) #5x5 sobel kernel

    for i in pics:
        if i[-5:] == '.jpeg':
            orignal_img = plt.imread(pic_path+i)

            plt.imshow(orignal_img)
            plt.axis('off')
            plt.show()

            img = orignal_img.mean(axis=-1)

            img_gx = convolve(G_x,img,[2,2,2,2],[1,1])
            img_gy = convolve(G_y,img,[2,2,2,2],[1,1])

            sq_img_gx = img_gx * img_gx
            sq_img_gy = img_gy * img_gy
            img_gx_gy = img_gx * img_gy

            # sq_img_gx = convolve(window, sq_img_gx, [2, 2, 2, 2], [1, 1])
            # sq_img_gy = convolve(window, sq_img_gy, [2, 2, 2, 2], [1, 1])
            # img_gx_gy = convolve(window, img_gx_gy, [2, 2, 2, 2], [1, 1])

            sq_img_gx = _2_dim_divided_convolve(linear_Gaussian_filter_5,sq_img_gx)
            sq_img_gy = _2_dim_divided_convolve(linear_Gaussian_filter_5,sq_img_gy)
            img_gx_gy = _2_dim_divided_convolve(linear_Gaussian_filter_5,img_gx_gy)

            score = score_for_each_pixel(sq_img_gx,sq_img_gy,img_gx_gy,0.05)
            final_img = Sign(orignal_img,score,[12,12])

            plt.imshow(final_img.astype(np.uint8))
            plt.axis('off')
            plt.show()


