import datetime

import cv2
import os
import sys
import numpy as np
import math
import glob
import pyspng
import PIL.Image

import torch
import lpips


def read_image(image_path):
    with open(image_path, 'rb') as f:
        if pyspng is not None and image_path.endswith('.png'):
            image = pyspng.load(f.read())
        else:
            image = np.array(PIL.Image.open(f))
    if image.ndim == 2:
        image = image[:, :, np.newaxis] # HW => HWC
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    image = image.transpose(2, 0, 1) # HWC => CHW
    image = torch.from_numpy(image).float().unsqueeze(0)
    image = image / 127.5 - 1

    return image


def calculate_metrics(folder1, folder2):
    l1 = sorted(glob.glob(folder1 + '/*.png') + glob.glob(folder1 + '/*.jpg'))
    l2 = sorted(glob.glob(folder2 + '/*.png') + glob.glob(folder2 + '/*.jpg'))
    assert(len(l1) == len(l2))
    # print('length:', len(l1))

    # l1 = l1[:3]; l2 = l2[:3];

    device = torch.device('cuda:0')
    loss_fn = lpips.LPIPS(net='alex').to(device)
    loss_fn.eval()
    # loss_fn = lpips.LPIPS(net='vgg').to(device)

    lpips_l = []
    with torch.no_grad():
        for i, (fpath1, fpath2) in enumerate(zip(l1, l2)):
            # print(i)
            _, name1 = os.path.split(fpath1)
            _, name2 = os.path.split(fpath2)
            name1 = name1.split('.')[0]
            name2 = name2.split('.')[0]
            assert name1 == name2, 'Illegal mapping: %s, %s' % (name1, name2)

            img1 = read_image(fpath1).to(device)
            img2 = read_image(fpath2).to(device)
            assert img1.shape == img2.shape, 'Illegal shape'
            lpips_l.append(loss_fn(img1, img2).mean().cpu().numpy())

    res = sum(lpips_l) / len(lpips_l)

    return res



if __name__ == '__main__':
    i = datetime.datetime.now()
    with open('lpips.txt', 'a+') as f:
        f.write("time:  " + str(i))
        f.write("\n")
    for i in range(0, 12):
        if i == 0:
            result = "result10"
        if i == 1:
            result = "result1"
        if i == 2:
            result = "result20"
        if i == 3:
            result = "result2"
        if i == 4:
            result = "result30"
        if i == 5:
            result = "result3"
        if i == 6:
            result = "result40"
        if i == 7:
            result = "result4"
        if i == 8:
            result = "result50"
        if i == 9:
            result = "result5"
        if i == 10:
            result = "result60"
        if i == 11:
            result = "result6"
        folder1 = os.path.join('E:/实验结果记录/不规则缺失/med/实验五/result_all_42_3', result, 'result')
    # folder1 = 'E:/实验结果记录/不规则缺失/med/实验五/result_all/result1/result'
        folder2 = 'E:/实验结果记录/不规则缺失/med/result_test_gt_256'
        res = calculate_metrics(folder1, folder2)
        print('lpips: %.4f' % res)
        with open('lpips.txt', 'a+') as f:
            f.write(result + ':   ')
            f.write('lpips: %.4f' % res)
            f.write('\n')
