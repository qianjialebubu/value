import os

import numpy as np
import math
import cv2

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def mse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return mse


def mae(img1, img2):
    mae = np.mean(abs(img1 - img2))

    return mae


def ssim(y_true, y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01 * 7)
    c2 = np.square(0.03 * 7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom
if __name__ == '__main__':
    gt_root = "E:/test/result_test_gt_256"
    pt_root = "E:/test/result"
    gt_file_name = os.listdir(gt_root)
    pt_file_name = os.listdir(pt_root)
    len = len(gt_root)
    mae_all = 0

    im1 = cv2.imread("E:/test/result_test_gt_256/0.png")
    im2 = cv2.imread("E:/test/result/0.png")
    mae = mae(im1, im2)
    print(mae)


    # for i in range(0,len):
    #     gt_full_name = os.path.join(gt_root,gt_file_name[i])
    #     pt_full_name = os.path.join(pt_root,pt_file_name[i])
    #     print(gt_full_name)
    #     print(pt_full_name)
    #     im1 = cv2.imread(gt_full_name)
    #     im2 = cv2.imread(pt_full_name)
    #     mae = mae(im1,im2)
    #     print(mae)
    #
    #     mae_all += mae
    # print(mae_all/len)

