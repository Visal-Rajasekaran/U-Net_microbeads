import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g) # up-conv feature map HxWxF_int
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 64, downsample=False) # 64 feature maps. 3 RGB channels colapsed
        self.enc2 = ConvBlock(64, 128, downsample=True) # 64 feature maps --> 128 feature maps. Map size 256x256
        self.enc3 = ConvBlock(128, 256, downsample=True) # 128 feature maps --> 256 feature maps. Map size 128x128
        self.enc4 = ConvBlock(256, 512, downsample=True) # 256 feature maps --> 512 feature maps. Map size 64x64

        self.bottleneck = ConvBlock(512, 1024, downsample=True)  # 512 --> 1024 feature maps. Map size 32x32

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # 1024 --> 512 feature maps. Map size 64x64
        self.att4 = AttentionBlock(512, 512, 256) # Masked
        self.dec4 = ConvBlock(1024, 512) # 512 feature maps from up-conv + 512 feature maps from skip connection concatenate to 1024
                                                               #  --> 512 feature maps. Map size 64x64

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # 512 --> 256 feature maps. Map size 128x128
        self.att3 = AttentionBlock(256, 256, 128) #
        self.dec3 = ConvBlock(512, 256) # 256 feature maps from up-conv + 256 feature maps from skip connection concatenate to 512
                                                              #  --> 256 feature maps. Map size 128x128

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # 256 --> 128 feature maps. Map size 256x256
        self.att2 = AttentionBlock(128, 128, 64)
        self.dec2 = ConvBlock(256, 128) # 128 feature maps from up-conv + 128 feature maps from skip connection concatenate to 256
                                                              #  --> 128 feature maps. Map size 256x256

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # 256 --> 128 feature maps. Map size 512x512
        self.att1 = AttentionBlock(64, 64, 32)
        self.dec1 = ConvBlock(128, 64) # 64 feature maps from up-conv + 64 feature maps from skip connection concatenate to 128
                                                             #  --> 64 feature maps. Map size 512x512

        # Outputs
        self.seg_out = nn.Conv2d(64, 1, kernel_size=1) # 64 feature maps --> 1 feature map. Output size is 512x512 grayscale



    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.bottleneck(s4)

        u4 = self.up4(b)
        a4 = self.att4(u4, s4)
        d4 = self.dec4(torch.cat([u4, a4], dim=1))

        u3 = self.up3(d4)
        a3 = self.att3(u3, s3)
        d3 = self.dec3(torch.cat([u3, a3], dim=1))

        u2 = self.up2(d3)
        a2 = self.att2(u2, s2)
        d2 = self.dec2(torch.cat([u2, a2], dim=1))

        u1 = self.up1(d2)
        a1 = self.att1(u1, s1)
        d1 = self.dec1(torch.cat([u1, a1], dim=1))

        seg = torch.sigmoid(self.seg_out(d1))

        return seg

class NANO_PICS(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_net = UNet(in_channels=3, out_channels=1)

    def forward(self, x):
        return self.shared_net(x)
