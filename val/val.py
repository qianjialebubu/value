import os
import datetime

from torch_fidelity import fidelity

from MAT_psnr.psnr3 import calculate_metrics
from lpips_test.cal_lpips import calculate_metrics as cl
def ssim_psnr_l1_lpips(txt,folder1_root,folder2):
    psnr_sum = 0
    ssim_sum = 0
    l1_sum = 0
    lpips_sum = 0
    i = datetime.datetime.now()
    txt = txt
    with open(txt, 'a+') as f:
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
        folder1 = os.path.join(folder1_root, result, 'result')
        psnr, ssim, dl1 = calculate_metrics(folder1, folder2)
        lpips = cl(folder1, folder2)
        psnr_sum+=psnr
        ssim_sum+=ssim
        l1_sum+=dl1
        lpips_sum+=lpips
        # print(psnr_sum, ssim_sum, l1_sum,lpips_sum)
        with open(txt, 'a+') as f:
            if(i%2==1):
                f.write(result + ':     ')
            else:
                f.write(result + ':    ')
            f.write('psnr: %.4f, ssim: %.4f, l1: %.4f, lpips: %.4f' % (psnr, ssim, dl1,lpips))
            f.write('\n')
    with open(txt, 'a+') as f:
        f.write('psnr_sum: %.4f, ssim_sum: %.4f, l1_sum: %.4f, lpips_sum: %.4f' % (float(psnr_sum/12), float(ssim_sum/12), float(l1_sum/12), float(lpips_sum/12)))
if __name__ == '__main__':
    txt = 'psnr_ssim_l1_lpips_med9.txt'
    folder2 = 'E:/test/result_test_gt_256'
    folder1 = 'E:/实验结果记录/不规则缺失/med/实验九/result_all'
    ssim_psnr_l1_lpips(txt=txt,folder2 = folder2 ,folder1_root= folder1)
    # fidelity  --fid --input1 E:\实验结果记录\不规则缺失\med\实验一\result_all\result40\result --input2 E:\test\result_test_gt_256
