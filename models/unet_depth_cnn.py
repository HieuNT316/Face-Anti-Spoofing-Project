# models/depth_cnn.py
import torch
import torch.nn as nn

class UNetDepthCNN(nn.Module):
    def __init__(self):
        super(UNetDepthCNN, self).__init__()

        # Encoder blocks
        self.enc1 = self.conv_block(3, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder blocks
        self.up4 = self.up_conv(512, 256)
        self.dec4 = self.conv_block(512, 256)  # skip + up

        self.up3 = self.up_conv(256, 128)
        self.dec3 = self.conv_block(256, 128)

        self.up2 = self.up_conv(128, 64)
        self.dec2 = self.conv_block(128, 64)

        self.up1 = self.up_conv(64, 32)
        self.dec1 = self.conv_block(64, 32)

        # Final conv to output depth map
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Maxpool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_ch, out_ch):
        return nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)        # 224x224
        e2 = self.enc2(self.pool(e1))  # 112x112
        e3 = self.enc3(self.pool(e2))  # 56x56
        e4 = self.enc4(self.pool(e3))  # 28x28

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # 14x14

        # Decoder
        d4 = self.up4(b)  # 14x14 -> 28x28
        d4 = torch.cat((e4, d4), dim=1)  # Skip
        d4 = self.dec4(d4)

        d3 = self.up3(d4)  # 28x28 -> 56x56
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)  # 56x56 -> 112x112
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)  # 112x112 -> 224x224
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        out = self.sigmoid(out)  # Chuẩn hoá 0-1

        return out

