import torch
import torch.nn as nn
import torch.nn.functional as F

class FGCN(nn.Module):
    def __init__(self, C_ch, W_ch):
        super(FGCN, self).__init__()
        self.W = W_ch
        self.dp = nn.Dropout(0.3)
        self.phi = nn.Conv2d(C_ch, W_ch, 1)
        self.relu_phi = nn.ReLU(inplace=True)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.rou = nn.Conv2d(C_ch, W_ch, 1)
        self.sig = nn.Sigmoid()

        self.LN2 = nn.LayerNorm(C_ch)
        self.relu = nn.ReLU(inplace=True)
        self.zeta = nn.Linear(C_ch, C_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        phi_xT = self.relu_phi(self.phi(x)).view(B, self.W, H * W)
        phi_x = phi_xT.transpose(1, 2)

        rou_x = self.sig(self.rou(self.global_pool(x))).view(B, self.W)
        diag = torch.diag_embed(rou_x)
        # A = torch.matmul(torch.matmul(phi_x,diag), phi_xT)

        D = torch.sum(torch.matmul(torch.matmul(phi_x, diag), phi_xT), dim=2)
        D_msqrt = torch.diag_embed(1 / (torch.sqrt(D + 1e-8)))
        # print(D_msqrt)
        P = torch.matmul(D_msqrt, phi_x)
        P_T = P.transpose(1, 2)

        x_trans = x.view(B, -1, C)
        LX = x_trans - torch.matmul(P, torch.matmul(diag, torch.matmul(P_T, x_trans)))

        output = self.relu(self.dp(self.zeta(LX)))
        return x+output.view(B, C, H, W)

class crossSpyGR_module(nn.Module):
    def __init__(self, C_ch, W_ch):
        super(crossSpyGR_module, self).__init__()
        self.W = W_ch
        self.dp = nn.Dropout(0.3)
        self.phi = nn.Conv2d(C_ch, W_ch, 1)
        self.relu_phi = nn.ReLU(inplace=True)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.rou = nn.Conv2d(C_ch, W_ch, 1)
        self.sig = nn.Sigmoid()

        self.LN2 = nn.LayerNorm(C_ch)
        self.relu = nn.ReLU(inplace=True)
        self.zeta = nn.Linear(C_ch, C_ch)  ####nn.Parameter(torch.randn(1, C_ch, C_ch))

    def forward(self, x, y):  ####xfor phi, yfor rou,x是主要的模态，y是控制的辅助模态
        # assert x.shape == y.shape

        B, C, H, W = x.shape
        phi_xT = self.relu_phi(self.phi(x)).view(B, self.W, H * W)
        phi_x = phi_xT.transpose(1, 2)

        rou_y = self.sig(self.rou(self.global_pool(y))).view(B, self.W)
        diag = torch.diag_embed(rou_y)
        # A = torch.matmul(torch.matmul(phi_x,diag), phi_xT)

        D = torch.sum(torch.matmul(torch.matmul(phi_x, diag), phi_xT), dim=2)
        Dy_msqrt = torch.diag_embed(1 / (torch.sqrt(D + 1e-8)))
        # print(Dy_msqrt)
        P = torch.matmul(Dy_msqrt, phi_x)
        P_T = P.transpose(1, 2)

        x_trans = x.view(B, -1, C)
        LX = x_trans - torch.matmul(P, torch.matmul(diag, torch.matmul(P_T, x_trans)))
        output = self.relu(self.dp(self.zeta(LX)))

        return x+output.view(B, C, H, W)

class deep_feafusion(nn.Module):
    def __init__(self, C_ch, W_ch):
        super(deep_feafusion, self).__init__()
        self.crossGR1 = crossSpyGR_module(C_ch, W_ch)

        self.crossGR2 = crossSpyGR_module(C_ch, W_ch)

    def forward(self, x, y):
        output_x = self.crossGR1(x, y)

        output_y = self.crossGR2(y, x)
        return output_x, output_y


class DepthWiseConv2d(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_size,stride,padding,bias=True):
        super(DepthWiseConv2d, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=bias,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    bias=bias)
    def forward(self, input):
        return self.point_conv(self.depth_conv(input))
class ConvBNReLU2d(nn.Module):
    def __init__(self, in_channels, out_channels,k=3,s=1,p=1):
        super(ConvBNReLU2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p ,stride=s, bias=False)
        self.BN = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.BN(x)
        return self.relu(x)
class DepConvBNReLU2d(nn.Module):
    def __init__(self, in_channels, out_channels,k=3,s=1,p=1,res=False):
        super(DepConvBNReLU2d, self).__init__()
        self.res = res
        self.conv1 = DepthWiseConv2d(in_channels, out_channels, kernel_size=k, padding=p ,stride=s, bias=False)
        self.BN = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.res:
            res = x
        x = self.conv1(x)
        x = self.BN(x)
        if self.res:
            return self.relu(x)+res
        else:
            return self.relu(x)

class SqueezeAndExcitation_x(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True),
                 weight=False):
        super(SqueezeAndExcitation_x, self).__init__()
        self.weight = weight
        self.fc1 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation)
        self.fc2 =nn.Sequential(
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )
        for m in self.fc2:
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
                # print(self.fc2[0].weight)

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x.detach(), 1)
        weighting = self.fc2(self.fc1(weighting)) + 0.5
        if not self.weight:
            return x * weighting
        else:
            return x * weighting, weighting

class SqueezeAndExcitation_cut(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True),
                 weight=False):
        super(SqueezeAndExcitation_cut, self).__init__()
        self.weight = weight
        self.fc1 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation)
        self.fc2 =nn.Sequential(
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )
        for m in self.fc2:
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
                # print(self.fc2[0].weight)

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x.detach(), 1)
        weighting = self.fc2(self.fc1(weighting)) - 0.5
        if not self.weight:
            return x * weighting
        else:
            return x * weighting, weighting
class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True), weight=False):
        super(SqueezeAndExciteFusionAdd, self).__init__()
        self.se_rgb = SqueezeAndExcitation_x(channels_in,
                                           activation=activation,weight=weight)
        self.se_depth = SqueezeAndExcitation_cut(channels_in,
                                             activation=activation,weight=weight)
    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        return rgb+depth     ###########

