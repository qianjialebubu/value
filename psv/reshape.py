import os
import cv2
x0=340
x1=596
y0=140
y1=140+256
img_path = "D:/file/work_space/deep_leaning/pythonProject1/psv/image/3.jpg"
save_path = "D:/file/work_space/deep_leaning/pythonProject1/psv/image/1.jpg"
img = cv2.imread(img_path)       #img_path为图片所在路径
crop_img = img[y0:y1,x0:x1]      #x0,y0为裁剪区域左上坐标；x1,y1为裁剪区域右下坐标
cv2.imwrite(save_path,crop_img)  #save_path为保存路径


# 批量修改文件名，默认操作为将图片按1，2，3，，，顺序重命名



path_in = "E:\data\巴黎街景\paris_train_original\paris_train_original3"  # 待批量重命名的文件夹
class_name = ".jpg"  # 重命名后的文件名后缀

file_in = os.listdir(path_in)  # 返回文件夹包含的所有文件名
num_file_in = len(file_in)  # 获取文件数目
# print(file_in, num_file_in)  # 输出修改前的文件名

for i in range(0, num_file_in):
    x0 = 340
    x1 = 596
    y0 = 140
    y1 = 140 + 256
    img_path = os.path.join(path_in,file_in[i])
    save_path =  os.path.join(path_in,file_in[i])
    t = str(i + 1)
    # new_name = os.rename(path_in + "/" + file_in[i], path_in + "/" + t + class_name)  # 重命名文件名
    print(img_path)
    img = cv2.imread(img_path)  # img_path为图片所在路径
    crop_img = img[y0:y1, x0:x1]  # x0,y0为裁剪区域左上坐标；x1,y1为裁剪区域右下坐标
    cv2.imwrite(save_path, crop_img)  # save_path为保存路径

file_out = os.listdir(path_in)
print(file_out)  # 输出修改后的结果