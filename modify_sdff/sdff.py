import torch
import torch.nn as nn
# v0.1 sdff模块的改造，使用结构和纹理一致的方法进行实验,增加了一个可学习的参数
# SK MODEL
#
# 把sdff的senet改为sknet
class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            # 使用不同kernel size的卷积
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features,
                              features,
                              kernel_size=3 + i * 2,
                              stride=stride,
                              padding=1 + i,
                              groups=G), nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False)))

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            print(i, fea_z.shape)
            vector = fc(fea_z).unsqueeze_(dim=1)
            print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

# SE MODEL
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.fc(y)
        return x * y.expand_as(x)


class SDFF(nn.Module):
    # Soft-gating Dual Feature Fusion.

    def __init__(self, in_channels, out_channels):
        super(SDFF, self).__init__()

        self.structure_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1),
            # SELayer(out_channels),
            SKConv(features=out_channels,WH=1, M=2, G=1, r=2),
            nn.Sigmoid()
        )
        self.detail_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1),
            # SELayer(out_channels),
            SKConv(features=out_channels, WH=1, M=2, G=1, r=2),
            nn.Sigmoid()
        )

        self.structure_gamma = nn.Parameter(torch.zeros(1))
        self.structure_beta = nn.Parameter(torch.zeros(1))
        self.detail_gamma = nn.Parameter(torch.zeros(1))

        self.detail_beta = nn.Parameter(torch.zeros(1))

# 前向传播函数中输入结构特征图和纹理特征图，即Fcst与Fcte。
    def forward(self, structure_feature, detail_feature):
        sd_cat = torch.cat((structure_feature, detail_feature), dim=1)

        map_detail = self.structure_branch(sd_cat)
        map_structure = self.detail_branch(sd_cat)

        detail_feature_branch = detail_feature + self.detail_beta * (structure_feature * (self.detail_gamma * (map_detail * detail_feature)))
        structure_feature_branch = structure_feature + self.structure_beta*(detail_feature*(self.structure_gamma * (map_structure * detail_feature)))

        return torch.cat((structure_feature_branch, detail_feature_branch), dim=1)




