import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

def normal_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def kaiming_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.constant_(m.weight.data, 1.0)
        torch.nn.init.constant_(m.bias.data, 0.0)

def weights_init(method):
    def init(m):
        if method == 'kaiming':
            kaiming_init(m)
        elif method == 'normal':
            normal_init(m)
        else:
            raise Exception("init method error")
    return init

def load_dataset(name,**kwargs):
    if name == 'MNIST':
        return load_MNIST(**kwargs)
    else:
        raise Exception("load dataset error")

def load_MNIST(batch_size,n_cpu,img_size,**kwargs):
    return DataLoader(datasets.MNIST('./images', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5], [0.5])
                                        ])),
                                        batch_size=batch_size, shuffle=True, num_workers=n_cpu)