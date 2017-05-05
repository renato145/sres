import torch
import torch.nn as nn
import torch.nn.parallel

def _weights_init( m):
    if isinstance(m, (nn.Conv2d)): 
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class SuperRes(nn.Module):
    def conv_block(self, main, name, inf, of, kernel, stride, padding, bn=True, act=True):
        main.add_module(f'{name}-{inf}.{of}.convt', nn.Conv2d(inf, of, kernel, stride, padding))
        if bn: main.add_module(f'{name}-{of}.batchnorm', nn.BatchNorm2d(of))
        if act: main.add_module(f'{name}-{of}.relu', nn.ReLU(inplace=True))
    
    def __init__(self, ngpu, upscale_factor=4):
        super(SuperRes, self).__init__()
        self.ngpu = ngpu
        
        model = nn.Sequential()
        self.conv_block(model, 'conv1', 3, 64, 5, 1, 2)
        self.conv_block(model, 'conv2', 64, 128, 3, 1, 1)
        self.conv_block(model, 'conv3', 128, 64, 3, 1, 1)
        self.conv_block(model, 'conv4', 64, 3 * (upscale_factor ** 2), 3, 1, 1, bn=False, act=False)
        
        model.add_module('pixshuffle', nn.PixelShuffle(upscale_factor))
        model.add_module(f'final.tanh', nn.Tanh())
        model.apply(_weights_init)
        self.model = model
    
    def forward(self, model_input):
        gpu_ids = None
        if isinstance(model_input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
            
        return nn.parallel.data_parallel(self.model, model_input, gpu_ids)
