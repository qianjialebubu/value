# function [MAE]=MAE_computating
# %本程序的功能是对显著性特征提取的结果计算Mean absolute error(MAE)值。
# %by hanlestudy@163.com
# clc
# clear
# imnames=dir(path_output);
# imnames2=dir(path_Targets);
# num=length(imnames);
# MAES=zeros(num,1);
# for j=1:num
#     Target=imread(imnames2(j).name); %读图
#     target=(Target)>0;        %二值化
#     Output=imread(imnames(j).name);
#     output=(Output)>0;
#     dis=abs(target-output);
#     MAES(j,1)=mean(mean(mean(dis)));
# end
# MAE=mean(MAES)
