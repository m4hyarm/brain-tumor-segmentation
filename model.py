import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ResidualUNet(nn.Module):
    """Residual UNet for medical image segmentation."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        features = init_features

        # Encoder
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.conv1x1_1 = nn.Conv2d(in_channels, features, kernel_size=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.conv1x1_2 = nn.Conv2d(features, features * 2, kernel_size=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.conv1x1_3 = nn.Conv2d(features * 2, features * 4, kernel_size=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.conv1x1_4 = nn.Conv2d(features * 4, features * 8, kernel_size=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")
        self.conv1x1_5 = nn.Conv2d(features * 8, features * 16, kernel_size=1)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, 2, stride=2)
        self.decoder4 = self._block(features * 16, features * 8, name="dec4")
        self.conv1x1_6 = nn.Conv2d(features * 16, features * 8, kernel_size=1)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 2, stride=2)
        self.decoder3 = self._block(features * 8, features * 4, name="dec3")
        self.conv1x1_7 = nn.Conv2d(features * 8, features * 4, kernel_size=1)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 2, stride=2)
        self.decoder2 = self._block(features * 4, features * 2, name="dec2")
        self.conv1x1_8 = nn.Conv2d(features * 4, features * 2, kernel_size=1)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")
        self.conv1x1_9 = nn.Conv2d(features * 2, features, kernel_size=1)

        self.conv_final = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = F.relu(self.encoder1(x) + self.conv1x1_1(x))
        enc2 = F.relu(self.encoder2(self.pool1(enc1)) + self.conv1x1_2(self.pool1(enc1)))
        enc3 = F.relu(self.encoder3(self.pool2(enc2)) + self.conv1x1_3(self.pool2(enc2)))
        enc4 = F.relu(self.encoder4(self.pool3(enc3)) + self.conv1x1_4(self.pool3(enc3)))

        # Bottleneck
        bottleneck = F.relu(self.bottleneck(self.pool4(enc4)) + self.conv1x1_5(self.pool4(enc4)))

        # Decoder
        dec4 = F.relu(self.decoder4(torch.cat((self.upconv4(bottleneck), enc4), dim=1)) + self.conv1x1_6(torch.cat((self.upconv4(bottleneck), enc4), dim=1)))
        dec3 = F.relu(self.decoder3(torch.cat((self.upconv3(dec4), enc3), dim=1)) + self.conv1x1_7(torch.cat((self.upconv3(dec4), enc3), dim=1)))
        dec2 = F.relu(self.decoder2(torch.cat((self.upconv2(dec3), enc2), dim=1)) + self.conv1x1_8(torch.cat((self.upconv2(dec3), enc2), dim=1)))
        dec1 = F.relu(self.decoder1(torch.cat((self.upconv1(dec2), enc1), dim=1)) + self.conv1x1_9(torch.cat((self.upconv1(dec2), enc1), dim=1)))

        return torch.sigmoid(self.conv_final(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(OrderedDict([
            (name + "_conv1", nn.Conv2d(in_channels, features, 3, padding=1, bias=False)),
            (name + "_bn1", nn.BatchNorm2d(features)),
            (name + "_relu1", nn.ReLU(inplace=True)),
            (name + "_conv2", nn.Conv2d(features, features, 3, padding=1, bias=False)),
            (name + "_bn2", nn.BatchNorm2d(features)),
        ]))
