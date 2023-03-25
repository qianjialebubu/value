import cv2
import math
import numpy

def psnr1(img1, img2):
    # compute mse
    # mse = np.mean((img1-img2)**2)
    mse = numpy.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr1 = 20 * math.log10(255 / math.sqrt(mse))
    return psnr1


def psnr2(img1, img2):
    mse = numpy.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1e-10:
        return 100
    psnr2 = 20 * math.log10(1 / math.sqrt(mse))
    return psnr2


imag2 = cv2.imread("E:\data\matlab\img\gt\000001.jpg")
print("imag1.shap: {}".format(imag2.shape))
imag1 = cv2.imread("E:\data\matlab\img\pt\000001.png")
print("imag1.shap: {}".format(imag1.shape))
image_size = [256, 256] #将图像转化为256*256大小的尺寸
imag1 = cv2.resize(imag1, image_size, interpolation=cv2.INTER_CUBIC)
imag1 = cv2.cvtColor(imag1, cv2.COLOR_BGR2GRAY)#将图像转化为灰度图像，不是必须转，也可以使用原始的彩色图像
imag2 = cv2.resize(imag2, image_size, interpolation=cv2.INTER_CUBIC)
imag2 = cv2.cvtColor(imag2, cv2.COLOR_BGR2GRAY)#将图像转化为灰度图像，不是必须转，也可以使用原始的彩色图像
res1 = psnr1(imag1, imag2)
print("res1:", res1)
res2 = psnr2(imag1, imag2)
print("res2:", res2)
