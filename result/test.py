import os
import datetime
# fidelity --gpu 0 --fid --input1 E:\实验结果记录\不规则缺失\med\实验一\result_all\result40\result --input2 E:\test\result_test_gt_256
from MAT_psnr.psnr3 import calculate_metrics

i = datetime.datetime.now()
with open('psnr_ssim_l3.txt', 'a+') as f:
    f.write("time:  " + str(i))
    f.write("\n")
for i in range(0, 12):
    if i == 0:
        result = "result1"
    if i == 1:
        result = "result10"
    if i == 2:
        result = "result2"
    if i == 3:
        result = "result20"
    if i == 4:
        result = "result3"
    if i == 5:
        result = "result30"
    if i == 6:
        result = "result4"
    if i == 7:
        result = "result40"
    if i == 8:
        result = "result5"
    if i == 9:
        result = "result50"
    if i == 10:
        result = "result6"
    if i == 11:
        result = "result60"
    folder1 = os.path.join('E:/实验结果记录/不规则缺失/med/实验一/全掩膜/result_all', result, 'result')
    # folder1 = 'E:/实验结果记录/不规则缺失/med/实验一/result_all/result40/result'
    folder2 = 'E:/test/result_test_gt_256'

    psnr, ssim, dl1 = calculate_metrics(folder1, folder2)
    with open('psnr_ssim_30.txt', 'a+') as f:
        f.write(result + ':   ')
        f.write('psnr: %.4f, ssim: %.4f, l1: %.4f' % (psnr, ssim, dl1))
        f.write('\n')

