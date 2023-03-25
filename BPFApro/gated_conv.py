import torch.nn as nn
import torch
#1.门控卷积的模块
class Gated_Conv(nn.Module):
    def __init__(self,in_ch,out_ch,ksize=3,stride=1,rate=1,activation=nn.ELU):
        super(Gated_Conv, self).__init__()
        padding=int(rate*(ksize-1)/2)
        #通过卷积将通道数变成输出两倍，其中一半用来做门控，学习
        self.conv=nn.Conv2d(in_ch,2*out_ch,kernel_size=ksize,stride=stride,padding=padding,dilation=rate)
        self.activation=activation
        self.relu = nn.ReLU

    def forward(self,x):
        raw=self.conv(x)

        x1=raw.split(int(raw.shape[1]/2),dim=1)#将特征图分成两半，其中一半是做学习
        gate=torch.sigmoid(x1[0])#将值限制在0-1之间
        print(type(gate))
        print(type(self.activation(x1[1])))
        out=self.activation(x1[1])*gate

        return out
if __name__ == '__main__':
    input = torch.ones((1,128,64,64))
    gate = Gated_Conv(in_ch=128,out_ch=128)
    output = gate(input)
    print(output.shape)