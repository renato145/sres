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
    
class SRES_GAN_D(nn.Module):
    def conv_block(self, main, name, inf, of, a, b, c, bn=True):
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        main.add_module(f'{name}-{inf}.{of}.conv', nn.Conv2d(inf, of, a, b, c, bias=False))
        main.add_module(f'{name}-{of}.batchnorm', nn.BatchNorm2d(of))
        main.add_module(f'{name}-{of}.relu', nn.LeakyReLU(0.2, inplace=True))

    def __init__(self, isize, nc, ndf, ngpu, n_extra_layers=0):
        super(SRES_GAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        self.conv_block(main, 'initial', nc, ndf, 4, 2, 1, False)
        csize, cndf = isize / 2, ndf
        for t in range(n_extra_layers):
            self.conv_block(main, f'extra-{t}', cndf, cndf, 3, 1, 1)
        
        idx = 1
        while csize > 4:
            self.conv_block(main, f'pyramid_{idx}', cndf, min(256, cndf*2), 4, 2, 1)
            cndf = min(256, cndf*2)
            csize /= 2
            idx += 1

        # state size. K x 4 x 4
        main.add_module(f'final.{cndf}-1.conv', nn.Conv2d(cndf, 1, 3, 1, 0, bias=False))
        self.main = main


    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        output = output.mean(0)
        return output.view(1)

class SRES_GAN_G(nn.Module):
    def __init__(self, img_size):
        super(SRES_GAN_G, self).__init__()

        self.initial = ConvBlock(3, 64, k=9, stride=1, padding=4, bias=False, bn=False)
        self.bottleneck_1 = Bottleneck(64, 64, 128)
        self.bottleneck_2 = Bottleneck(128, 128, 256)
        # self.bottleneck_3 = Bottleneck(256, 128, 128)
        self.conv_1 = ConvBlock(256, 64, k=3, stride=1, padding=1, bias=False, act=False)
        self.pixel_shuffle_1 = PixelShuffleBlock(64, 128)
        self.pixel_shuffle_2 = PixelShuffleBlock(128, 128)
        self.pixel_shuffle_3 = PixelShuffleBlock(128, 128)
        self.conv_out = ConvBlock(128, 3, k=9, stride=1, padding=4, bias=False, bn=False, act=False)
        # self.conv_out = ConvBlock(64, 3, k=9, stride=1, padding=4, bias=False, bn=False, act=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        residual = self.initial(x)
        out = self.bottleneck_1(residual)
        out = self.bottleneck_2(out)
        # out = self.bottleneck_3(out)
        out = self.conv_1(out)
        out += residual
        out = self.pixel_shuffle_1(out)
        out = self.pixel_shuffle_2(out)
        out = self.pixel_shuffle_3(out)
        out = self.conv_out(out)
        out = self.tanh(out)

        return out