import math
import random
import time
import pdb

import PIL
import pyspng
from options.test_options import TestOptions
from data.dataprocess import DataProcess
from models.models import create_model
import torchvision
from torch.utils import data
#from torch.utils.tensorboard import SummaryWriter
import cv2
import torch
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
import torchvision.transforms as transforms
import os
import datetime

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def calculate_l1(img1, img2):
    img1 = img1.astype(np.float64) / 255.0
    img2 = img2.astype(np.float64) / 255.0
    l1 = np.mean(np.abs(img1 - img2))

    return l1


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
    # image = image.transpose(2, 0, 1) # HWC => CHW

    return image


def calculate_metrics(folder1, folder2):
    l1 = sorted(glob.glob(folder1 + '/*.png') + glob.glob(folder1 + '/*.jpg'))
    l2 = sorted(glob.glob(folder2 + '/*.png') + glob.glob(folder2 + '/*.jpg'))
    assert(len(l1) == len(l2))
    print('length:', len(l1))

    # l1 = l1[:3]; l2 = l2[:3];

    psnr_l, ssim_l, dl1_l = [], [], []
    for i, (fpath1, fpath2) in enumerate(zip(l1, l2)):
        # print(i)
        # print(fpath1)
        # print(fpath2)
        _, name1 = os.path.split(fpath1)
        _, name2 = os.path.split(fpath2)
        name1 = name1.split('.')[0]
        name2 = name2.split('.')[0]
        assert name1 == name2, 'Illegal mapping: %s, %s' % (name1, name2)

        img1 = read_image(fpath1).astype(np.float64)
        img2 = read_image(fpath2).astype(np.float64)
        assert img1.shape == img2.shape, 'Illegal shape'
        psnr_l.append(calculate_psnr(img1, img2))
        ssim_l.append(calculate_ssim(img1, img2))
        dl1_l.append(calculate_l1(img1, img2))

    psnr = sum(psnr_l) / len(psnr_l)
    ssim = sum(ssim_l) / len(ssim_l)
    dl1 = sum(dl1_l) / len(dl1_l)

    return psnr, ssim, dl1

if __name__ == "__main__":
    for epoch in range(30,41,2):

        img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])


        opt = TestOptions().parse()
        model = create_model(opt)
        model.netEN.module.load_state_dict(torch.load("/data3/qianjiale_dataset/project/MED/MED3/checkpoints/Mutual Encoder-Decoder/"+str(epoch)+"_net_EN.pth")['net'])
        model.netDE.module.load_state_dict(torch.load("/data3/qianjiale_dataset/project/MED/MED3/checkpoints/Mutual Encoder-Decoder/"+str(epoch)+"_net_DE.pth")['net'])
        model.netMEDFE.module.load_state_dict(torch.load("/data3/qianjiale_dataset/project/MED/MED3/checkpoints/Mutual Encoder-Decoder/"+str(epoch)+"_net_MEDFE.pth")['net'])
        mask_paths = glob('{:s}/*'.format(opt.mask_root))
        de_paths = glob('{:s}/*'.format(opt.de_root))
        st_path = glob('{:s}/*'.format(opt.st_root))
        mask_len = len(mask_paths)
        image_len = len(de_paths )
        for j in range(12):
            index = j*1000+999
            # print(index)
            if j == 0:
                results_dir = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result1/result'
                results_dir_mask = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result1/result_mask'
                results_dir_test_gt = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result1/result_test_gt'
                mask_path = r'/data3/qianjiale_dataset/testing_mask/testing_mask_dataset_10_fn'
            if j == 1:
                results_dir = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result10/result'
                results_dir_mask = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result10/result_mask'
                results_dir_test_gt = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result10/result_test_gt'
                mask_path = r'/data3/qianjiale_dataset/testing_mask/testing_mask_dataset_10_f'
            if j == 2:
                results_dir = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result2/result'
                results_dir_mask = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result2/result_mask'
                results_dir_test_gt = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result2/result_test_gt'
                mask_path = r'/data3/qianjiale_dataset/testing_mask/testing_mask_dataset_20_fn'
            if j == 3:
                results_dir = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result20/result'
                results_dir_mask = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result20/result_mask'
                results_dir_test_gt = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result20/result_test_gt'
                mask_path = r'/data3/qianjiale_dataset/testing_mask/testing_mask_dataset_20_f'
            if j == 4:
                results_dir = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result3/result'
                results_dir_mask = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result3/result_mask'
                results_dir_test_gt = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result3/result_test_gt'
                mask_path = r'/data3/qianjiale_dataset/testing_mask/testing_mask_dataset_30_fn'
            if j == 5:
                results_dir = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result30/result'
                results_dir_mask = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result30/result_mask'
                results_dir_test_gt = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result30/result_test_gt'
                mask_path = r'/data3/qianjiale_dataset/testing_mask/testing_mask_dataset_30_f'
            if j == 6:
                results_dir = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result4/result'
                results_dir_mask = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result4/result_mask'
                results_dir_test_gt = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result4/result_test_gt'
                mask_path = r'/data3/qianjiale_dataset/testing_mask/testing_mask_dataset_40_fn'
            if j == 7:
                results_dir = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result40/result'
                results_dir_mask = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result40/result_mask'
                results_dir_test_gt = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result40/result_test_gt'
                mask_path = r'/data3/qianjiale_dataset/testing_mask/testing_mask_dataset_40_f'
            if j == 8:
                results_dir = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result5/result'
                results_dir_mask = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result5/result_mask'
                results_dir_test_gt = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result5/result_test_gt'
                mask_path = r'/data3/qianjiale_dataset/testing_mask/testing_mask_dataset_50_fn'
            if j == 9:
                results_dir = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result50/result'
                results_dir_mask = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result50/result_mask'
                results_dir_test_gt = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result50/result_test_gt'
                mask_path = r'/data3/qianjiale_dataset/testing_mask/testing_mask_dataset_50_f'
            if j == 10:
                results_dir = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result6/result'
                results_dir_mask = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result6/result_mask'
                results_dir_test_gt = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result6/result_test_gt'
                mask_path = r'/data3/qianjiale_dataset/testing_mask/testing_mask_dataset_60_fn'
            if j == 11:
                results_dir = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result60/result'
                results_dir_mask = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result60/result_mask'
                results_dir_test_gt = r'/data3/qianjiale_dataset/project/MED/MED3/result_all/result60/result_test_gt'
                mask_path = r'/data3/qianjiale_dataset/testing_mask/testing_mask_dataset_60_f'

            for i in tqdm(range(image_len)):
                # only use one mask for all image
                # path_m = mask_paths[0]
                mask_pathss = glob('{:s}/*'.format(mask_path))
                mask_path_len = glob('{:s}/*'.format(mask_path))
                len_1 = len(mask_path_len)
                path_m = mask_pathss[random.randint(0,len_1-1)]
                # print(len)
                # print(path_m)
                path_d = de_paths[i]
                path_s = de_paths[i]

                mask = Image.open(path_m).convert("RGB")
                detail = Image.open(path_d).convert("RGB")
                structure = Image.open(path_s).convert("RGB")
                detail_save = detail
                mask_save = mask
                mask = mask_transform(mask)
                detail = img_transform(detail)
                structure = img_transform(structure)
                mask = torch.unsqueeze(mask, 0)
                detail = torch.unsqueeze(detail, 0)
                structure = torch.unsqueeze(structure,0)

                with torch.no_grad():
                    model.set_input(detail, structure, mask)
                    model.forward()
                    fake_out = model.fake_out
                    fake_out = fake_out.detach().cpu() * mask + detail*(1-mask)
                    fake_image = (fake_out+1)/2.0
                output = fake_image.detach().numpy()[0].transpose((1, 2, 0))*255
                output = Image.fromarray(output.astype(np.uint8))
                output.save(rf"{results_dir}/{i}.png")
                mask_save.save(rf"{results_dir_mask}/{i}.png")
                detail_save.save(rf"{results_dir_test_gt}/{i}.png")


        # 使用函数返回测试结果

        i = datetime.datetime.now()
        with open('psnr_ssim_30.txt', 'a+') as f:
            f.write("time:  " + str(i)+"    epoch=" +str(epoch))
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
            folder1 = os.path.join('/data3/qianjiale_dataset/project/MED/MED3/result_all', result, 'result')
            # folder1 = 'E:/实验结果记录/不规则缺失/med/实验一/result_all/result40/result'
            folder2 = 'E:/test/result_test_gt_256'
            psnr, ssim, dl1 = calculate_metrics(folder1, folder2)
            with open('psnr_ssim_30.txt', 'a+') as f:
                f.write(result + ':   ')
                f.write('psnr: %.4f, ssim: %.4f, l1: %.4f' % (psnr, ssim, dl1))
                f.write('\n')
