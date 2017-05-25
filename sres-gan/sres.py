import os
import cv2
import torch
import numpy as np
from PIL import Image
from model import SuperRes
from torch import FloatTensor as FT
from torch.autograd import Variable
from torchvision import transforms

class SRES(object):
    def __init__(self, path='model_weights.pkl', n_gpu=0):
        self.path = path
        self.n_gpu = n_gpu
        self.load_net()
        self.to_np = lambda x: np.transpose((x/2+0.5).clamp(0,1).numpy(), (1,2,0))

        
    def load_net(self):
        self.net = SuperRes(self.n_gpu, 4)
        if self.n_gpu > 0:
            self.net = self.net.cuda()
            data = torch.load(self.path)
        else:
            data = torch.load(self.path, map_location=lambda storage, loc: storage)
            
        self.net.load_state_dict(data['model'])
        self.net.eval()
    
    def Var(self, *params):
        if self.n_gpu > 0:
            return Variable(FT(*params).cuda())
        else:
            return Variable(FT(*params))
    
    def apply_path(self, img_path, resize_inp=None):
        transform_layers = [transforms.Scale(resize_inp)] if resize_inp else []
        transform_layers += [transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform = transforms.Compose(transform_layers)
        img = Image.open(img_path).convert('RGB')
        img = self.Var(transform(img).unsqueeze_(0))
        img_out = self.net(img).data.cpu()
        img_out = self.to_np(img_out[0])
        
        return img_out
    
    def apply_cv2(self, img):
        transform = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.Var(transform(img).unsqueeze_(0))
        img_out = self.net(img).data.cpu()
        img_out = self.to_np(img_out[0])
        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        
        return img_out
        