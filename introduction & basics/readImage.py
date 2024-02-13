#coding:utf-8
import matplotlib.pyplot as plt

if __name__=='__main__':
    img = plt.imread('./1.jpeg')
    
    #显示图片
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    #观察像素的表示方式
    print(img[10,300])
    
    #灰度化
    img = img.mean(axis = -1)
    plt.imshow(img,cmap='gray')
    plt.axis('off')
    plt.show()
