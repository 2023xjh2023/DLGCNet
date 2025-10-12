import glob
import os
import numpy as np
import cv2
from PIL import Image
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu
from torchvision.transforms import (Pad, ColorJitter, Resize, FiveCrop, RandomResizedCrop,
                                    RandomHorizontalFlip, RandomRotation, RandomVerticalFlip)
import random
from matplotlib import pyplot as plt

###############################potsdam#######################################################################
# img_paths_raw = glob.glob(os.path.join('/wuyi/UDA_dataset/Potsdam/test_images', "*.tif")) #train  test
# nDSM_paths_raw = glob.glob(os.path.join('/wuyi/UDA_dataset/Potsdam/test_nDSM', "*.jpg"))  #train  test
# img_paths_raw.sort()##train key=lambda name: (name[51:53],int(name[53:-8]))
# nDSM_paths_raw.sort()                #train key=lambda name: name[50:-22]
# save_path='/wuyi/UDA_dataset/Potsdam/test_RGBnDSM/top_potsdam_'  #train  test
# for img_path, nDSM_path in zip(img_paths_raw, nDSM_paths_raw):
#     print(img_path)
#     print(nDSM_path)
#     #/wuyi/UDA_dataset/Potsdam/train_images/top_potsdam_7_7_RGB.tif
#     img_path2=save_path+img_path[50:-8]+'_RGBnDSM.tif' #51   50
#     print(img_path2)
#     print('------------------------')
#     img1 = cv2.imread(img_path)
#     img2 = cv2.imread(nDSM_path,cv2.IMREAD_GRAYSCALE)
#     if not img2.shape[1] == 6000:
#         img2=cv2.resize(img2, (6000,6000))
#     b_1, g_1, r_1 = cv2.split(img1)  # 这里需要知道的是split是按照RBG进行通道的分离
#     # 会报错提示too many values to unpack (expected x)
#     res = cv2.merge([b_1, g_1, r_1, img2])  #
#
#     cv2.imwrite(img_path2, res,((int(cv2.IMWRITE_TIFF_RESUNIT), 2,int(cv2.IMWRITE_TIFF_COMPRESSION), 1,
#                                                                   int(cv2.IMWRITE_TIFF_XDPI), 100,
#                                                                   int(cv2.IMWRITE_TIFF_YDPI), 100)))
#
# AA=cv2.imread('/wuyi/UDA_dataset/Potsdam/test_RGBnDSM/top_potsdam_6_15_RGBnDSM.tif',cv2.IMREAD_UNCHANGED) #train 2_10
# print(AA.shape)

########################################################################################################################
# AA=cv2.imread('/wuyi/UDA_dataset/Potsdam/train/images_RGBnDSM512/top_potsdam_2_10_0_0.tif',cv2.IMREAD_UNCHANGED)
# #train 2_10
# print(AA.shape)
# b_1, g_1, r_1,a= cv2.split(AA)  # 这里需要知道的是split是按照RBG进行通道的分离
# cv2.imwrite('/wuyi/UDA_dataset/Potsdam/train/images_RGBnDSM512/aaaaaaaa.tif', a,((int(cv2.IMWRITE_TIFF_RESUNIT), 2,
#                                                                    int(cv2.IMWRITE_TIFF_COMPRESSION), 1,
#                                                                    int(cv2.IMWRITE_TIFF_XDPI), 100,
#                                                                    int(cv2.IMWRITE_TIFF_YDPI), 100)))

# AA=cv2.imread('/wuyi/UDA_dataset/Potsdam/test/images_RGBnDSM512/top_potsdam_6_15_0_0.tif',cv2.IMREAD_UNCHANGED) #train 2_10
# print(AA.shape)
# b_1, g_1, r_1,a= cv2.split(AA)  # 这里需要知道的是split是按照RBG进行通道的分离
# cv2.imwrite('/wuyi/UDA_dataset/Potsdam/test/images_RGBnDSM512/aaaaaaaa.tif', a,((int(cv2.IMWRITE_TIFF_RESUNIT), 2,
#                                                                    int(cv2.IMWRITE_TIFF_COMPRESSION), 1,
#                                                                    int(cv2.IMWRITE_TIFF_XDPI), 100,
#                                                                    int(cv2.IMWRITE_TIFF_YDPI), 100)))

######################################vaihingen########################################################################
# img_paths_raw = glob.glob(os.path.join('/wuyi/UDA_dataset/Vaihingen/train_images', "*.tif")) #train
# nDSM_paths_raw = glob.glob(os.path.join('/wuyi/UDA_dataset/Vaihingen/train_nDSM', "*.jpg"))  #
# img_paths_raw.sort(key=lambda name: int(name[61:-4]))   #train61 test60
# nDSM_paths_raw.sort(key=lambda name: int(name[61:-15])) #train61 test60
# save_path='/wuyi/UDA_dataset/Vaihingen/train_IRRGnDSM/top_mosaic_09cm_'  #train
# for img_path, nDSM_path in zip(img_paths_raw, nDSM_paths_raw):
#     print(img_path)
#     print(nDSM_path)
#     img_path2=save_path+img_path[57:-4]+'.tif' #train57 test56
#     print(img_path2)
#     print('------------------------------------')
#     img1 = cv2.imread(img_path)
#     img2 = cv2.imread(nDSM_path,cv2.IMREAD_GRAYSCALE)
#     if not img2.shape[1] == img1.shape[1] and img2.shape[0] == img1.shape[0]:
#         img2=cv2.resize(img2, (img1.shape[0],img1.shape[1]))
#     b_1, g_1, r_1 = cv2.split(img1)  # 这里需要知道的是split是按照RBG进行通道的分离
#     res = cv2.merge([b_1, g_1, r_1, img2])  #
#
#     cv2.imwrite(img_path2, res,((int(cv2.IMWRITE_TIFF_RESUNIT), 2,int(cv2.IMWRITE_TIFF_COMPRESSION), 1,
#                                                                   int(cv2.IMWRITE_TIFF_XDPI), 100,
#                                                                   int(cv2.IMWRITE_TIFF_YDPI), 100)))
####################################################################################################################
# AA=cv2.imread('/wuyi/UDA_dataset/Vaihingen/train/images_nDSM512/top_mosaic_09cm_area1_0_16.tif',cv2.IMREAD_UNCHANGED)
#
# print(AA.shape)
# b_1, g_1, r_1,a= cv2.split(AA)  # 这里需要知道的是split是按照RBG进行通道的分离
# cv2.imwrite('/wuyi/UDA_dataset/Vaihingen/train/images_nDSM512/aaaaaaaa.tif', a,((int(cv2.IMWRITE_TIFF_RESUNIT), 2,
#                                                                    int(cv2.IMWRITE_TIFF_COMPRESSION), 1,
#                                                                    int(cv2.IMWRITE_TIFF_XDPI), 100,
#                                                                    int(cv2.IMWRITE_TIFF_YDPI), 100)))
####################################################################################################################
# AA=cv2.imread('/wuyi/UDA_dataset/Vaihingen/test/images_nDSM512/top_mosaic_09cm_area30_0_16.tif',cv2.IMREAD_UNCHANGED)
# #train 2_10
# print(AA.shape)
# b_1, g_1, r_1,a= cv2.split(AA)  # 这里需要知道的是split是按照RBG进行通道的分离
# cv2.imwrite('/wuyi/UDA_dataset/Vaihingen/test/images_nDSM512/aaaaaaaa.tif', a,((int(cv2.IMWRITE_TIFF_RESUNIT), 2,
#                                                                    int(cv2.IMWRITE_TIFF_COMPRESSION), 1,
#                                                                    int(cv2.IMWRITE_TIFF_XDPI), 100,
#                                                                    int(cv2.IMWRITE_TIFF_YDPI), 100)))
###################################################################################################################


# data_path = '/wuyi/UDA_dataset/Vaihingen/nDSM/normalized_DSMs'  # 数据集路径
# img_height, img_width = 6000, 6000  # 图片尺寸
#
# # 初始化变量
# img_count = 0
# img_sum = np.zeros((1,))
# img_squared_sum = np.zeros((1,))
#
# # 遍历数据集文件夹中的所有图片
# for root, dirs, files in os.walk(data_path):
#     for file in files:
#         img_path = os.path.join(root, file)
#         img = cv2.imread(img_path,0).astype(np.float32) / 255
#         img = cv2.resize(img, (img_width, img_height))  # 缩放图片至指定尺寸
#         img_count += 1
#         img_sum += np.sum(img, axis=(0, 1))
#         img_squared_sum += np.sum(img ** 2, axis=(0, 1))
#
# # 计算每个通道的均值和标准差
# channel_mean = img_sum / (img_count * img_height * img_width)
# channel_std = np.sqrt((img_squared_sum / (img_count * img_height * img_width)) - (channel_mean ** 2))
#
# print('每个通道的均值：', channel_mean)
# print('每个通道的标准差：', channel_std)

# print('mean', (0.20179997 * 8 / 38 + 0.17244304 * 30 / 38))
# print('std', np.sqrt(0.23874217 * 0.23874217 *8/30+ 0.20814824 * 0.20814824 * 30 / 38))
############################################################################################
#/wuyi/UDA_dataset/Potsdam/train/masks_RGBnDSM512/top_potsdam_2_10_0_0.png
#/wuyi/UDA_dataset/Potsdam/test/masks_RGBnDSM512/top_potsdam_6_15_0_0.png
#/wuyi/UDA_dataset/Vaihingen/train/masks_nDSM512/top_mosaic_09cm_area1_0_8.png
#/wuyi/UDA_dataset/Vaihingen/test/masks_nDSM512/top_mosaic_09cm_area30_0_8.png
img1 = cv2.imread('/wuyi/UDA_dataset/Potsdam/test/masks_RGBnDSM512/top_potsdam_6_15_0_0.png', -1)
print(np.in1d([0,1,2,3,4,5,6], img1))
print(img1)
print(img1.shape)
###############################################################################
# img1 = cv2.imread('/wuyi/UDA_dataset/Potsdam/train/masks_RGBnDSM512/top_potsdam_2_10_0_0.png', 0)
# # 别忘了中括号 [img],[0],None,[256],[0,256]，只有 mask 没有中括号
# hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
# img2 = cv2.imread('/wuyi/UDA_dataset/Potsdam/train/masks_RGBnDSM512/top_potsdam_2_10_0_0.png')
# color = ('b', 'g', 'r')
# for i, col in enumerate(color):
#     histr = cv2.calcHist([img2], [i], None, [256], [0, 256])
#     plt.subplot(224), plt.plot(histr, color=col),
#     plt.xlim([0, 16]), plt.title('Histogram')
#
# plt.subplot(221), plt.imshow(img1, 'gray'), plt.title('Image1')
# plt.subplot(222), plt.hist(img1.ravel(), 256, [0, 256]),
# plt.title('Histogram'), plt.xlim([0, 16])
# plt.subplot(223), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)), plt.title('Image2')
# plt.show()



###############################################whu-opt-sar-rgb#########################################################
# img_paths_raw = glob.glob(os.path.join('/wuyi/遥感数据集/whu-opt-sar/optical', "*.tif")) #train
# # nDSM_paths_raw = glob.glob(os.path.join('/wuyi/UDA_dataset/Vaihingen/train_nDSM', "*.jpg"))  #
# # img_paths_raw.sort(key=lambda name: int(name[61:-4]))   #train61 test60
# # nDSM_paths_raw.sort(key=lambda name: int(name[61:-15])) #train61 test60
# save_path='/wuyi/遥感数据集/whu-opt-sar/rgb/'  #train
# if not os.path.isdir('/wuyi/遥感数据集/whu-opt-sar/rgb'):
#     os.mkdir('/wuyi/遥感数据集/whu-opt-sar/rgb')
# for img_path in img_paths_raw:
#     print(img_path)
#     img_path2 = save_path+img_path[32:]
#     print(img_path2)
#     print('------------------------------------')
#     img1 = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
#     # img2 = cv2.imread(nDSM_path,cv2.IMREAD_GRAYSCALE)
#     # if not img2.shape[1] == img1.shape[1] and img2.shape[0] == img1.shape[0]:
#     #     img2=cv2.resize(img2, (img1.shape[0],img1.shape[1]))
#     b_1, g_1, r_1 ,ir= cv2.split(img1)  # 这里需要知道的是split是按照RBG进行通道的分离
#     res = cv2.merge([r_1, g_1, b_1])  #img2(对于whu-opt-sar，好像原本就是按照BGR-IR通道排的？)
#     # print(ir)
#
#     cv2.imwrite(img_path2, res,((int(cv2.IMWRITE_TIFF_RESUNIT), 2,int(cv2.IMWRITE_TIFF_COMPRESSION), 1,
#                                                                   int(cv2.IMWRITE_TIFF_XDPI), 100,
#                                                                   int(cv2.IMWRITE_TIFF_YDPI), 100)))
#############################################whu-opt-sar-ir#######################################################
# img_paths_raw = glob.glob(os.path.join('/wuyi/遥感数据集/whu-opt-sar/optical', "*.tif")) #train
# # nDSM_paths_raw = glob.glob(os.path.join('/wuyi/UDA_dataset/Vaihingen/train_nDSM', "*.jpg"))  #
# # img_paths_raw.sort(key=lambda name: int(name[61:-4]))   #train61 test60
# # nDSM_paths_raw.sort(key=lambda name: int(name[61:-15])) #train61 test60
# save_path='/wuyi/遥感数据集/whu-opt-sar/nir/'  #train
# if not os.path.isdir('/wuyi/遥感数据集/whu-opt-sar/nir'):
#     os.mkdir('/wuyi/遥感数据集/whu-opt-sar/nir')
# for img_path in img_paths_raw:
#     print(img_path)
#     img_path2 = save_path+img_path[32:]
#     print(img_path2)
#     print('------------------------------------')
#     img1 = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
#     # img2 = cv2.imread(nDSM_path,cv2.IMREAD_GRAYSCALE)
#     # if not img2.shape[1] == img1.shape[1] and img2.shape[0] == img1.shape[0]:
#     #     img2=cv2.resize(img2, (img1.shape[0],img1.shape[1]))
#     b_1, g_1, r_1 ,ir= cv2.split(img1)  # 这里需要知道的是split是按照RBG进行通道的分离
#     res = cv2.merge([r_1, g_1, b_1])  #img2(对于whu-opt-sar，好像原本就是按照BGR-IR通道排的？)
#     # print(ir)
#
#     cv2.imwrite(img_path2, ir,((int(cv2.IMWRITE_TIFF_RESUNIT), 2,int(cv2.IMWRITE_TIFF_COMPRESSION), 1,
#                                                                   int(cv2.IMWRITE_TIFF_XDPI), 100,
#                                                                   int(cv2.IMWRITE_TIFF_YDPI), 100)))
