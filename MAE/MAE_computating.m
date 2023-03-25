function [MAE]=MAE_computating
%本程序的功能是对显著性特征提取的结果计算Mean absolute error(MAE)值。
%by hanlestudy@163.com
clc
clear
path_output = 'E:\test\result\';
path_Targets = 'E:\test\result_test_gt_256\';


path_output_list = dir(strcat(path_output,'*.png'));
path_Targets_list = dir(strcat(path_Targets,'*.png'));


len=length(path_output_list);  
%imnames = path_output_list;
%imnames2 = path_Targets_list;
%imnames2=dir(path_Targets_list);  
num=length(len);
MAES=zeros(num,1);
for j=1:num
    Target_name = path_Targets_list(j).name;
    output_name = path_output_list(j).name;
    %fprintf(output_name);
    %fprintf(path_Targets_list);
    %fprintf(strcat(path_Targets_list,Target_name));
    Target = imread(strcat(path_Targets,Target_name));
    Output = imread(strcat(path_output,output_name));
    %Target=imread(strcat(path_Targets_list,imnames2(j).name)); %读图
    target=(Target)>0;        %二值化
    %Output=imread(strcat(path_output_list,imnames(j).name));
    output=(Output)>0;
    dis=abs(target-output);
    MAES(j,1)=mean(mean(mean(dis)));
    
end
MAE=mean(MAES) ;

