import torch
import torch.nn as nn
import torch.nn.parallel

class ConvBlock(nn.Module):
    def __init__(self, in_f, out_f, k=3, stride=1, padding=1, bias=True, bn=True, act=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_f, out_f, k, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_f)
        self.relu = nn.ReLU(inplace=True)
        self.do_bn = bn
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        if self.do_bn:
            out = self.bn(out)
        if self.act:
            out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    def __init__(self, in_f, mid_f, out_f):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvBlock(in_f, mid_f, k=1, padding=0, bias=False)
        self.conv2 = ConvBlock(mid_f, out_f, k=3, bias=False)
        self.conv3 = ConvBlock(in_f, out_f, k=1, padding=0, bias=False)

    def forward(self, x):
        residual = x
        residual = self.conv3(residual)

        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual

        return out

class PixelShuffleBlock(nn.Module):
    def __init__(self, in_f, out_f, k=3, stride=1, padding=1, bias=False):
        super(PixelShuffleBlock, self).__init__()
        self.conv = nn.Conv2d(in_f, out_f * 4, k, stride, padding, bias=bias)
        self.pshuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.pshuffle(out)
        out = self.relu(out)

        return out

class DiscBlock(nn.Module):
    def __init__(self, in_f, out_f, k=3, stride=1, padding=1, bias=False, bn=True, act=True):
        super(DiscBlock, self).__init__()
        self.conv = nn.Conv2d(in_f, out_f, k, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_f)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.do_bn = bn
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        if self.do_bn:
            out = self.bn(out)
        if self.act:
            out = self.lrelu(out)

        return out

class SRES_GAN_D(nn.Module):
    def __init__(self):
        super(SRES_GAN_D, self).__init__()

        self.disc_block1 = DiscBlock(3, 32)
        self.disc_block2 = DiscBlock(32, 64, 4, 2)
        self.disc_block3 = DiscBlock(64, 128, 4, 2)
        self.disc_block4 = DiscBlock(128, 128, 4, 2)
        self.disc_block5 = DiscBlock(128, 128, 4, 2)
        self.disc_block6 = DiscBlock(128, 128, 4, 2)
        self.disc_block7 = DiscBlock(128, 64, 4, 2)
        self.disc_block8 = DiscBlock(64, 32, 4, 2)
        self.disc_block9 = DiscBlock(32, 1, 2, 2, bn=False, act=False)

    def forward(self, x):
        out = self.disc_block1(x)
        out = self.disc_block2(out)
        out = self.disc_block3(out)
        out = self.disc_block4(out)
        out = self.disc_block5(out)
        out = self.disc_block6(out)
        out = self.disc_block7(out)
        out = self.disc_block8(out)
        out = self.disc_block9(out)

        return out

class SRES_GAN_G(nn.Module):
    def __init__(self):
        super(SRES_GAN_G, self).__init__()

        self.initial = ConvBlock(3, 64, k=9, stride=1, padding=4, bias=False, bn=False)
        self.bottleneck_1 = Bottleneck(64, 64, 128)
        self.bottleneck_2 = Bottleneck(128, 128, 256)
        self.bottleneck_3 = Bottleneck(256, 128, 128)
        self.conv_1 = ConvBlock(128, 64, k=3, stride=1, padding=1, bias=False, act=False)
        self.pixel_shuffle_1 = PixelShuffleBlock(64, 128)
        self.pixel_shuffle_2 = PixelShuffleBlock(128, 128)
        # self.pixel_shuffle_3 = PixelShuffleBlock(128, 128)
        self.conv_out = ConvBlock(128, 3, k=9, stride=1, padding=4, bias=False, bn=False, act=False)
        # self.conv_out = ConvBlock(64, 3, k=9, stride=1, padding=4, bias=False, bn=False, act=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        residual = self.initial(x)
        out = self.bottleneck_1(residual)
        out = self.bottleneck_2(out)
        out = self.bottleneck_3(out)
        out = self.conv_1(out)
        out += residual
        out = self.pixel_shuffle_1(out)
        out = self.pixel_shuffle_2(out)
        # out = self.pixel_shuffle_3(out)
        out = self.conv_out(out)
        out = self.tanh(out)

        return out
    