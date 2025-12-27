import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g   : channels of gating signal (from deeper decoder)
            F_l   : channels of skip connection (encoder)
            F_int : intermediate channels
        """
        super(AttentionBlock, self).__init__()

        # Gating signal projection (no downsampling, just 1x1 conv)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Skip connection projection (downsample to match g's spatial size)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Combine and produce attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g : gating signal from deeper decoder (lower res)
        x : skip connection from encoder (higher res)
        """
        g1 = self.W_g(g)   # shape: [B, F_int, H/2, W/2]
        x1 = self.W_x(x)   # shape: [B, F_int, H/2, W/2]

        psi = self.relu(g1 + x1)  # combine
        psi = self.psi(psi)       # [B, 1, H/2, W/2]

        # Upsample attention map back to skip connection resolution
        psi = F.interpolate(psi, size=x.shape[2:], mode='bilinear', align_corners=True)

        # Apply attention coefficients to the original skip connection
        return x * psi


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder (ConvBlock then Pool)
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(512, 1024)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # Attention: g from bottleneck (1024), x from enc4 (512)
        self.att4 = AttentionBlock(F_g=1024, F_l=512, F_int=256)
        self.dec4 = ConvBlock(512 + 512, 512)   # concat(u4(512), a4(512)) -> 1024 -> convblock -> 512

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # Attention: g from d4 (512), x from enc3 (256)
        self.att3 = AttentionBlock(F_g=512, F_l=256, F_int=128)
        self.dec3 = ConvBlock(256 + 256, 256)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Attention: g from d3 (256), x from enc2 (128)
        self.att2 = AttentionBlock(F_g=256, F_l=128, F_int=64)
        self.dec2 = ConvBlock(128 + 128, 128)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Attention: g from d2 (128), x from enc1 (64)
        self.att1 = AttentionBlock(F_g=128, F_l=64, F_int=32)
        self.dec1 = ConvBlock(64 + 64, 64)

        # Final output conv
        self.seg_out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder: produce skip feature maps s1..s4, and pooled features p1..p4
        s1 = self.enc1(x)          # size: H x W, channels 64
        p1 = self.pool1(s1)        # H/2 x W/2

        s2 = self.enc2(p1)         # H/2 x W/2, channels 128
        p2 = self.pool2(s2)        # H/4 x W/4

        s3 = self.enc3(p2)         # H/4 x W/4, channels 256
        p3 = self.pool3(s3)        # H/8 x W/8

        s4 = self.enc4(p3)         # H/8 x W/8, channels 512
        p4 = self.pool4(s4)        # H/16 x W/16

        # Bottleneck at coarse resolution
        b = self.bottleneck(p4)    # H/16 x W/16, channels 1024

        # Decoder:
        #  * For gating s4, use g = b (coarser). AttentionBlock will project g and upsample projected g to s4 size.
        u4 = self.up4(b)           # upsample b -> spatial H/8 x W/8, channels 512
        a4 = self.att4(b, s4)      # gate s4 using b (coarse)
        d4 = self.dec4(torch.cat([u4, a4], dim=1))  # channels -> 512 out

        # For gating s3, use g = d4 (one level deeper)
        u3 = self.up3(d4)          # H/4 x W/4, channels 256
        a3 = self.att3(d4, s3)     # gate s3 using d4 (coarser)
        d3 = self.dec3(torch.cat([u3, a3], dim=1))  # -> 256 out

        # For gating s2, use g = d3
        u2 = self.up2(d3)          # H/2 x W/2, channels 128
        a2 = self.att2(d3, s2)
        d2 = self.dec2(torch.cat([u2, a2], dim=1))  # -> 128 out

        # For gating s1, use g = d2
        u1 = self.up1(d2)          # H x W, channels 64
        a1 = self.att1(d2, s1)
        d1 = self.dec1(torch.cat([u1, a1], dim=1))  # -> 64 out

        seg = torch.sigmoid(self.seg_out(d1))
        return seg

# wrapper same as before
class NANO_PICS(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_net = UNet(in_channels=3, out_channels=1)

    def forward(self, x):
        return self.shared_net(x)
