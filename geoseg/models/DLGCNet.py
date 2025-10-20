import torch
import torch.nn as nn
from .MPT import RVSA_MTP

from .GCFF import FGCN, deep_feafusion, DepConvBNReLU2d, ConvBNReLU2d, SqueezeAndExciteFusionAdd
from .GCFF import DepthWiseConv2d

class addlayer(nn.Module):
    def __init__(self, **kwargs):
        super(addlayer, self).__init__(**kwargs)

    def forward(self, x,y):
        return x+y

class DLGCNet(nn.Module):
    def __init__(self, C_ch=512, W_ch=64, num_class=6):
        super(DLGCNet, self).__init__()
        self.MTP = RVSA_MTP(
            img_size=512,
            patch_size=16,
            drop_path_rate=0.3,
            out_indices=[3, 5, 7, 11],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=False,
            use_abs_pos_emb=True,
            pretrained='/checkpoints/last_vit_b_rvsa_ss_is_rd_pretrn_model_encoder.pth',
            D2LoRA_r = 4,#4,#9,##4
            D2LoRA_alpha = 4,#4,#9,##4
        )
        self.MTP.init_weights()

        self.reduceC_3c = DepthWiseConv2d(768, C_ch, 3, 2, 1, bias=True)
        ##############layer for nDSM or SAR
        self.layer0 = nn.Sequential(DepConvBNReLU2d(1, 128, 7, 2, 3),
                                    nn.MaxPool2d(2, 2),
                                    DepConvBNReLU2d(128, 128, 3, 1, 1, True)
                                    )
        self.trans_l0 = nn.Sequential(nn.Upsample(size=32, mode='bilinear'),
                                    ConvBNReLU2d(128, 768, 1, 1, 0)
                                      )

        self.layer1 = nn.Sequential(DepConvBNReLU2d(128, 256, 3, 2, 1),
                                    DepConvBNReLU2d(256, 256, 3, 1, 1, True)
                                    )
        self.trans_l1 = nn.Sequential(nn.Upsample(size=32, mode='bilinear'),
                                      ConvBNReLU2d(256, 768, 1, 1, 0)
                                      )

        self.layer2 = nn.Sequential(DepConvBNReLU2d(256, 512, 3, 2, 1),
                                    DepConvBNReLU2d(512, 512, 3, 1, 1, True)
                                    )
        self.trans_l2 = nn.Sequential(nn.Upsample(size=32, mode='bilinear'),
                                      ConvBNReLU2d(512, 768, 1, 1, 0)
                                      )

        self.layer3 = nn.Sequential(DepConvBNReLU2d(512, 1024, 3, 2, 1),
                                    DepConvBNReLU2d(1024, 1024, 3, 1, 1, True)
                                    )
        self.trans_l3 = nn.Sequential(nn.Upsample(size=32, mode='bilinear'),
                                      ConvBNReLU2d(1024, 768, 1, 1, 0)
                                      )

        self.reduceC_nDSM = DepthWiseConv2d(1024, C_ch, 3, 1, 1, bias=True)
        #############################################
        self.MFI0 = SqueezeAndExciteFusionAdd(channels_in=768)
        self.MFI1 = SqueezeAndExciteFusionAdd(channels_in=768)
        self.MFI2 = SqueezeAndExciteFusionAdd(channels_in=768)
        self.MFI3 = SqueezeAndExciteFusionAdd(channels_in=768)
        ######################################GCFF####################################
        self.FE_vis = nn.Sequential(FGCN(C_ch, W_ch), FGCN(C_ch, W_ch))
        self.FE_D = nn.Sequential(FGCN(C_ch, W_ch), FGCN(C_ch, W_ch))
        self.DFF1 = deep_feafusion(C_ch, W_ch)
        self.DFF2 = deep_feafusion(C_ch, W_ch)
        self.DFFE_V = nn.Sequential(FGCN(C_ch, W_ch), FGCN(C_ch, W_ch))
        self.DFFE_D = nn.Sequential(FGCN(C_ch, W_ch), FGCN(C_ch, W_ch))
        self.add = addlayer()
        #############################decoder########################################
        self.decoder3 = nn.Sequential(DepConvBNReLU2d(512 + 768, 512),
                                      DepConvBNReLU2d(512, 512, 3, 1, 1, True))
        self.decoder2 = nn.Sequential(DepConvBNReLU2d(512 + 768, 256),
                                      DepConvBNReLU2d(256, 256, 3, 1, 1, True))
        self.decoder1 = nn.Sequential(DepConvBNReLU2d(256 + 768, 128),
                                      DepConvBNReLU2d(128, 128, 3, 1, 1, True))

        self.seghead = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=4),
                                     # DepthWiseConv2d(128, num_class, 3, 1, 1, bias=False))
                                     nn.Conv2d(128, num_class, 3, padding=1, bias=False))

    def forward(self, x_VIS, x_nDSM):
        B = x_VIS.shape[0]
        x_VIS, (Hp, Wp) = self.MTP.patch_embed(x_VIS)
        x_VIS = self.MTP.pos_drop(x_VIS + self.MTP.pos_embed)

        for i in range(3):
            if i == 0:
                FV0 = self.MTP.blocks[i](x_VIS, Hp, Wp)
            else:
                FV0 = self.MTP.blocks[i](FV0, Hp, Wp)
        FD0 = self.layer0(x_nDSM.unsqueeze(1))
        FF0 = self.MFI0(FV0.permute(0, 2, 1).reshape(B, -1, Hp, Wp), self.trans_l0(FD0))  ##[B,768,32,32]
        # print(FF0.shape)

        for i in range(3,6):
            if i == 3:
                FV1 = self.MTP.blocks[i](FF0.flatten(2).permute(0, 2, 1), Hp, Wp)
            else:
                FV1 = self.MTP.blocks[i](FV1, Hp, Wp)
        FD1 = self.layer1(FD0)
        FF1 = self.MFI1(FV1.permute(0, 2, 1).reshape(B, -1, Hp, Wp), self.trans_l1(FD1))

        for i in range(6, 9):
            if i == 6:
                FV2 = self.MTP.blocks[i](FF1.flatten(2).permute(0, 2, 1), Hp, Wp)
            else:
                FV2 = self.MTP.blocks[i](FV2, Hp, Wp)
        FD2 = self.layer2(FD1)
        FF2 = self.MFI2(FV2.permute(0, 2, 1).reshape(B, -1, Hp, Wp), self.trans_l2(FD2))

        for i in range(9, 12):
            if i == 9:
                FV3 = self.MTP.blocks[i](FF2.flatten(2).permute(0, 2, 1), Hp, Wp)
            else:
                FV3 = self.MTP.blocks[i](FV3, Hp, Wp)
        FD3 = self.layer3(FD2)
        FF3 = self.MFI3(FV3.permute(0, 2, 1).reshape(B, -1, Hp, Wp), self.trans_l3(FD3))

        F_enhance_V = self.FE_vis(self.reduceC_3c(FF3))
        F_enhance_D = self.FE_D(self.reduceC_nDSM(FD3))

        DFFV1, DFFD1 = self.DFF1(F_enhance_V, F_enhance_D)
        DFFV2, DFFD2 = self.DFF2(DFFV1, DFFD1)

        DFFEV = self.DFFE_V(DFFV2)
        DFFED = self.DFFE_D(DFFD2)

        add = self.add(DFFEV,DFFED)
        fea_dec3 = self.decoder3(torch.cat([nn.Upsample(scale_factor=2, mode='bilinear')(add)
                                                                      ,FF2],
                                                                      dim=1))

        fea_dec2 = self.decoder2(torch.cat([nn.Upsample(scale_factor=2,mode='bilinear')(fea_dec3),
                                            nn.Upsample(scale_factor=2,mode='bilinear')(FF1)],
                                           dim=1))

        fea_dec1 = self.decoder1(torch.cat([nn.Upsample(scale_factor=2, mode='bilinear')(fea_dec2),
                                            nn.Upsample(scale_factor=4, mode='bilinear')(FF0)],
                                           dim=1))

        return  self.seghead(fea_dec1)



if __name__ == "__main__":
    model = DLGCNet()
    x = torch.randn(4,3,512,512)
    y = torch.randn(4,512,512)
    print(model(x,y).shape)
