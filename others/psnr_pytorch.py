import cv2
import numpy as np
import math


def psnr1(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def psnr2(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__ == '__main__':
    gt = cv2.imread('E:\data\matlab\img\gt\000001.jpg')
    img = cv2.imread('E:\data\matlab\img\pt\000001.jpg')
    print(psnr1(gt, img))
    print(psnr2(gt, img))