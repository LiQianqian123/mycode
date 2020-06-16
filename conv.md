```
import numpy as np 
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
#%matplotlib inline
#卷积模块
#卷积在 pytorch 中有两种方式，一种是 torch.nn.Conv2d()，一种是 torch.nn.functional.conv2d()
#这两种形式的卷积对于输入的要求都是一样的，首先需要输入是一个 torch.autograd.Variable() 的类型，大小是 (batch, channel, H, W)，其中 batch 表示输入的一批数据的数目，第二个是输入的通道数，一般一张彩色的图片是 3，灰度图是 1，而卷积网络过程中的通道数比较大，会出现几十到几百的通道数，H 和 W 表示输入图片的高度和宽度，比如一个 batch 是 32 张图片，每张图片是 3 通道，高和宽分别是 50 和 100，那么输入的大小就是 (32, 3, 50, 100)

im = Image.open('/home/videostudy/liqianqian/picture_processing/imgpng/dogs.png').convert('L')  #读入灰度图
im = np.array(im,dtype = 'float32')      #将图片变成数组
print(im)
#将图片矩阵转化成tensor
im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))    #(batch,channel,H,W)  
print(im.shape)
##使用conv2d,使用f.conv2d
#定义一个算子对其进行轮廓检测
conv1 = nn.Conv2d(1,1,3,bias=False)     #conv2d(in_channels,out_channels,kernel_size,stride,padding,bias,groups)
sobel_kernel = np.array([[-1, -1,-1],[-1,8,-1],[-1,-1,-1]],dtype='float32')     #定义轮廓检测算子
#print(sobel_kernel.shape)   #(3,3)
sobel_kernel = sobel_kernel.reshape((1,1,3,3))    
conv1.weight.data = torch.from_numpy(sobel_kernel)    #给卷积的kernel赋值

edge1 = conv1(Variable(im))
#edge1 = edge1.data.squeeze().numpy()      #将输出其转化成图片的格式，numpy三维，需要降一个维度(channel,H,W)
#plt.imshow(edge1, cmap='gray')            #想用图片显示数据类型必须是numpy

##使用f.conv
sobel_kernel = np.array([[-1, -1,-1],[-1,8,-1],[-1,-1,-1]],dtype = 'float32')
sobel_kernel = sobel_kernel.reshape((1,1,3,3))
weigth = Variable(np.from_numpy(sobel_kernel))
edge1 = F.conv2d(Variable(im),weight)


#池化模块   常用的池化有：nn.MaxPool2d, nn.AvgPool(2d)
#在 pytorch 中最大值池化的方式也有两种，一种是 nn.MaxPool2d()，一种是 torch.nn.functional.max_pool2d()
##使用nn.MaxPool2d：要先定义网络
pool1 = nn.MaxPool2d(2,2)    #nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices )
print('before max pool, image shape:{} x {}'.format(im.shape[2],im.shape[3]))    #为什么是[2][3],参考：im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1]))) 
small_im1 = pool1(Variable(im))
#small_im1 = small_im1.data.squeeze().numpy()
print('after max pool, image shape:{} x {}'.format(small_im1.shape[0],small_im1.shape[1]))
#plt.imshow(small_im1.cmap = gray)
#结果：池化层只是减小了图片的尺寸，并不会影响图片的内容

##使用F.maxpool2d:相当于只是个函数
print('before max pool, image shape: {} x {}'.format(im.shape[2], im.shape[3]))
small_im2 = F.max_pool2d(Variable(im), 2, 2)
#small_im2 = small_im2.data.squeeze().numpy()
print('after max pool, image shape: {} x {} '.format(small_im1.shape[0], small_im1.shape[1]))
#plt.imshow(small_im1.cmap = gray)
```
