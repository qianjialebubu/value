import torch
import torch.nn as nn
import torch.nn.functional as F
class GlobalContextBlock(nn.Module):
    def __init__(self, in_channels, scale = 16):
        super(GlobalContextBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels//scale

        self.Conv_key = nn.Conv2d(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(dim=1)

        self.Conv_value = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.in_channels, 1),
        )
        self.Conv_Y = nn.Conv2d(self.in_channels,self.in_channels
                                ,kernel_size=7,stride=1,padding=6,dilation=2)

    def forward(self, x):
        b, c, h, w = x.size()
        # key -> [b, 1, H, W] -> [b, 1, H*W] ->  [b, H*W, 1]
        key = self.SoftMax(self.Conv_key(x).view(b, 1, -1).permute(0, 2, 1).view(b, -1, 1).contiguous())
        query = x.view(b, c, h*w)
        # [b, c, h*w] * [b, H*W, 1]
        concate_QK = torch.matmul(query, key)
        concate_QK = concate_QK.view(b, c, 1, 1).contiguous()
        value = self.Conv_value(concate_QK)
        out = x + value
        out = self.Conv_Y(out)
        return out

if __name__ == "__main__":
    x = torch.randn((2, 1024, 124, 124))
    GCBlock = GlobalContextBlock(in_channels=1024)
    out = GCBlock(x)

    print("GCBlock  input.shape:", x.shape)
    print("GCBlock output.shape:", out.shape)
    # print(out)
# 对于3*3的卷积核，使用扩张卷积
# self.Conv_Y = nn.Conv2d(self.in_channels,self.in_channels
#                                 ,kernel_size=3,stride=1,padding=2,dilation=2)
# 对于5*5的卷积核，使用扩张卷积
# self.Conv_Y = nn.Conv2d(self.in_channels,self.in_channels
#                                 ,kernel_size=5,stride=1,padding=4,dilation=2)
# 对于7*7的卷积核，使用扩张卷积
# self.Conv_Y = nn.Conv2d(self.in_channels,self.in_channels
#                                 ,kernel_size=7,stride=1,padding=6,dilation=2)
# 使用扩张卷积要求输出与输入的形状不变，要求卷积核的padding = kernel_size-1