# 批量修改文件名，默认操作为将图片按1，2，3，，，顺序重命名

import os

path_in = "E:\data\celeba-256-test-PT\GT"  # 待批量重命名的文件夹
file_in = os.listdir(path_in)  # 返回文件夹包含的所有文件名
num_file_in = len(file_in)  # 获取文件数目
print(num_file_in)
m=0
for i in range(0, num_file_in):
    m +=1
print(m)