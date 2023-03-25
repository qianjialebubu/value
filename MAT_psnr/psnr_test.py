import os
import datetime

from psnr3 import calculate_metrics

i = datetime.datetime.now()
with open('psnr_ssim_test.txt', 'a+') as f:
    f.write("time:  " + str(i))
    f.write("\n")
for i in range(0, 12):
    if i == 0:
        result = "result10"
        folder1 = os.path.join('E:/实验结果记录/不规则缺失/med/实验四/result_all/result10/result')
    if i == 1:
        result = "result1"
        folder1 = os.path.join('E:/实验结果记录/不规则缺失/med/实验四/result_all/result1/result')
    if i == 2:
        result = "result20"
        folder1 = os.path.join('E:/实验结果记录/不规则缺失/med/实验四/result_all/result20/result')
    if i == 3:
        result = "result2"
        folder1 = os.path.join('E:/实验结果记录/不规则缺失/med/实验四/result_all/result2/result')
    if i == 4:
        result = "result30"
        folder1 = os.path.join('E:/实验结果记录/不规则缺失/med/实验四/result_all/result30/result')
    if i == 5:
        result = "result3"
        folder1 = os.path.join('E:/实验结果记录/不规则缺失/med/实验四/result_all/result3/result')
    if i == 6:
        result = "result40"
        folder1 = os.path.join('E:/实验结果记录/不规则缺失/med/实验四/result_all/result40/result')
    if i == 7:
        result = "result4"
        folder1 = os.path.join('E:/实验结果记录/不规则缺失/med/实验四/result_all/result4/result')
    if i == 8:
        result = "result50"
        folder1 = os.path.join('E:/实验结果记录/不规则缺失/med/实验四/result_all/result50/result')
    if i == 9:
        result = "result5"
        folder1 = os.path.join('E:/实验结果记录/不规则缺失/med/实验四/result_all/result5/result')
    if i == 10:
        result = "result60"
        folder1 = os.path.join('E:/实验结果记录/不规则缺失/med/实验四/result_all/result60/result')
    if i == 11:
        result = "result6"
        folder1 = os.path.join('E:/实验结果记录/不规则缺失/med/实验四/result_all/result6/result')
    # folder1 = 'E:/实验结果记录/不规则缺失/med/实验一/result_all/result40/result'
    folder2 = 'E:/test/result_test_gt_256'
    psnr, ssim, dl1 = calculate_metrics(folder1, folder2)
    with open('psnr_ssim_test.txt', 'a+') as f:
        f.write(result + ':   ')
        f.write('psnr: %.4f, ssim: %.4f, l1: %.4f' % (psnr, ssim, dl1))
        f.write('\n')