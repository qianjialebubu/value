import os

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
import cv2
pt_root = 'E:/test/result'
gt_root =  'E:/test/result_test_gt_256'
pt_list = []
gt_list = []
ssim_count = 0
psnr_count = 0
mse_count = 0
def getSSIM(img1, img2):
    return compare_ssim(img1, img2, multichannel=True)  # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True


def getPSNR(img1, img2):
    return compare_psnr(img1, img2)


def getMSE(img1, img2):
    return compare_mse(img1, img2)


if __name__ == '__main__':
    class_name_pt = os.listdir(pt_root)
    class_name_gt = os.listdir(gt_root)
    for name in class_name_pt:
        pt_list.append(name)
    for name in class_name_gt:
        gt_list.append(name)
    len = len(pt_list)
    print(len)
    for i in range(len):
        img1_root = os.path.join(gt_root,gt_list[i])
        img2_root = os.path.join(pt_root,pt_list[i])

        img1 = cv2.imread(img1_root)
        img2 = cv2.imread(img2_root)
        ssim = getSSIM(img1, img2)
        psnr = getPSNR(img1, img2)
        mse = getMSE(img1, img2)
        ssim_count = ssim_count +ssim
        psnr_count = psnr_count +psnr
        mse_count = mse_count + mse

    print("ssim="+str(ssim_count/len))
    print("mse="+str(mse_count/len))
    print("psnr="+str(psnr_count/len))
