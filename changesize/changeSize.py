import os

import numpy as np
import cv2 as cv
img_root = "E:/test/mask/result_mask"
img_root_out = "E:/test/mask/mask_256"
file_in = os.listdir(img_root)
len = len(file_in)
print(len)
for i in range(0,len):
    name= file_in[i]
    full_name = os.path.join(img_root,name)
    img = cv.imread(full_name)
    # 缩放图像，后面的其他程序都是在这一行上改动
    dst = cv.resize(img, (256, 256))
    # 显示图像
    # cv.imshow("dst: %d x %d" % (dst.shape[0], dst.shape[1]), dst)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # 输出图像
    full_name_out = os.path.join(img_root_out,name)
    cv.imwrite(full_name_out,dst)
    print(i)

