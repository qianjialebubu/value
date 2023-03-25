import cv2
import numpy as np

img_a = cv2.imread("D:/file/work_space/deep_leaning/pythonProject1/add_image/img_gt/37.png")

img_b = cv2.imread("D:/file/work_space/deep_leaning/pythonProject1/add_image/img_mask/37.png")
img_c = cv2.imread("D:/file/work_space/deep_leaning/pythonProject1/add_image/img_gt/37.png")
# img_b = cv2.resize(img_b,256,256)
# xishu = 1
# print(img_b[137][1][0])
for i in range(255):
    for j in range(255):
        if(img_b[i][j][0] ==255 ):
            img_c[i][j] = [255,255,255]
# img_a = cv2.resize(img_a, (img_b.shape[1], img_b.shape[0]))

# img_c = xishu * img_b + (1 - xishu) * img_a
img_c = img_c.astype(np.uint8)
img_c = np.clip(img_c, 0, 255)
cv2.imshow('asdf', img_c)


cv2.waitKey()
cv2.imwrite("D:/file/work_space/deep_leaning/pythonProject1/add_image/result/37.png",img_c)