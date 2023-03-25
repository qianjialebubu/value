from cv2 import imread
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

I = imread('E:\data\matlab\img\gt\000004.jpg');
J = imread('E:\data\matlab\img\pt\000004.jpg_fake_B.png');
PSNR = peak_signal_noise_ratio(I, J)

