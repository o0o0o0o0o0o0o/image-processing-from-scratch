# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def rawFFT(Img, Wr,axis):
    if Img.shape[axis] == 2:
        pic = np.zeros(Img.shape, dtype=complex)
        if axis == 1:
            pic[:,0] = Img[:,0] + Img[:,1] * Wr[:,0]
            pic[:,1] = Img[:,0] - Img[:,1] * Wr[:,0]
        elif axis == 0:
            pic[0,:] = Img[0,:] + Img[1,:] * Wr[0,:]
            pic[1,:] = Img[0,:] - Img[1,:] * Wr[0,:]
        return pic
    else:
        pic = np.empty(Img.shape, dtype=complex)
        if axis == 1:
            A = rawFFT(Img[:,::2], Wr[:,::2],axis)
            B = rawFFT(Img[:,1::2], Wr[:,::2],axis)
            pic[:,0:Img.shape[1] // 2] = A + Wr * B
            pic[:,Img.shape[1] // 2:Img.shape[1]] = A - Wr * B
        elif axis == 0:
            A = rawFFT(Img[::2,:], Wr[::2,:], axis)
            B = rawFFT(Img[1::2,:], Wr[::2,:], axis)
            pic[0:Img.shape[0] // 2,:] = A + Wr * B
            pic[Img.shape[0] // 2:Img.shape[0],:] = A - Wr * B
        return pic


def FFT_1d(Img,axis):
    Wr = np.zeros(Img.shape,dtype=complex)
    if axis == 0:
        Wr = np.zeros((Img.shape[0]//2,Img.shape[1]), dtype=complex)
        temp = np.array([
            np.cos(2*np.pi*i/Img.shape[0]) - 1j*np.sin(2*np.pi*i/Img.shape[0]) for i in range(Img.shape[0]//2)])
        for i in range(Wr.shape[1]):
            Wr[:,i] = temp
    elif axis == 1:
        Wr = np.zeros((Img.shape[0], Img.shape[1]//2), dtype=complex)
        temp = np.array([
            np.cos(2 * np.pi * i / Img.shape[1]) - 1j * np.sin(2 * np.pi * i / Img.shape[1]) for i in
            range(Img.shape[1] // 2)])
        for i in range(Wr.shape[0]):
            Wr[i,:] = temp
    return rawFFT(Img, Wr,axis)

def iFFT_1d(Img,axis):
    Wr = np.zeros(Img.shape,dtype=complex)
    if axis == 0:
        Wr = np.zeros((Img.shape[0]//2,Img.shape[1]), dtype=complex)
        temp = np.array([
            np.cos(2*np.pi*i/Img.shape[0]) + 1j*np.sin(2*np.pi*i/Img.shape[0]) for i in range(Img.shape[0]//2)])
        for i in range(Wr.shape[1]):
            Wr[:,i] = temp
    elif axis == 1:
        Wr = np.zeros((Img.shape[0], Img.shape[1]//2), dtype=complex)
        temp = np.array([
            np.cos(2 * np.pi * i / Img.shape[1]) + 1j * np.sin(2 * np.pi * i / Img.shape[1]) for i in
            range(Img.shape[1] // 2)])
        for i in range(Wr.shape[0]):
            Wr[i,:] = temp

    return rawFFT(Img, Wr,axis)*(1.0/Img.shape[axis])


def FFT_2d(Img):
    '''
    only for gray scale 2d-img. otherwise return 0 img with the same size of Img
    :param Img: img to be fourior transform
    :return: img been transformed
    '''
    imgsize = Img.shape
    pic = np.zeros(imgsize, dtype=complex)
    if len(imgsize) == 2:
        N = 2
        while N < imgsize[0]:
            N = N << 1
        num1 = N - imgsize[0]

        N = 2
        while N < imgsize[1]:
            N = N << 1
        num2 = N - imgsize[1]

        pic = FFT_1d(np.pad(Img, ((num1 // 2, num1 - num1 // 2), (0, 0)), 'edge'), 0)[
              num1 // 2:num1 // 2 + imgsize[0], :]
        pic = FFT_1d(np.pad(pic, ((0, 0), (num2 // 2, num2 - num2 // 2)), 'edge'), 1)[:,
              num2 // 2:num2 // 2 + imgsize[1]]

    return pic


def iFFT_2d(Img):
    '''
    only for gray scale 2d-img. otherwise return 0 img with the same size of Img
    :param Img: img to be fourior transform
    :return: img been transformed
    '''
    imgsize = Img.shape
    pic = np.zeros(imgsize, dtype=complex)
    if len(imgsize) == 2:
        N = 2
        while N < imgsize[0]:
            N = N << 1
        num1 = N - imgsize[0]

        N = 2
        while N < imgsize[1]:
            N = N << 1
        num2 = N - imgsize[1]

        pic = iFFT_1d(np.pad(Img,((num1//2,num1-num1//2),(0,0)),'edge'),0)[num1//2:num1//2+imgsize[0],:]  # ,constant_values=(255,255)
        pic = iFFT_1d(np.pad(pic,((0,0),(num2//2,num2-num2//2)),'edge'),1)[:,num2//2:num2//2+imgsize[1]]  # ,constant_values=(255,255)

    return pic


if __name__ == "__main__":

    img = plt.imread('./img/1.jpg')
    img = img.mean(2)       # gray


    plt.imshow(img.astype(np.uint8),cmap='gray')
    plt.axis('off')
    plt.show()

    F1 = np.fft.fft2(img[:256, :256])
    F2 = FFT_2d(img[:256, :256])
    print((abs(F1 - F2) < 0.0000001).all())

    F1 = np.fft.ifft2(F1)
    F2 = iFFT_2d(F2)
    print((abs(F1 - F2) < 0.0000001).all())

    F2 = np.abs(F2)
    F2[F2 > 255] = 255

    plt.imshow(F2.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()

    F1 = np.abs(F1)
    F1[F1 > 255] = 255

    plt.imshow(F1.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()