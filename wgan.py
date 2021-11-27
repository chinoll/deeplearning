import torch.nn as nn
import torch.nn.functional as F
import torch
import visdom

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from utils import *
import os
import numpy as np
import math

from config import *
os.makedirs("images",exist_ok=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.size = img_size//4
        self.fc1 = nn.Linear(latent_dim, 128*self.size**2)

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        out = self.fc1(x)
        out = out.view(out.shape[0], 128, self.size, self.size)
        out = self.conv(out)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=False):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size//2**4
        self.adv_layer = nn.Sequential(nn.Linear(128*ds_size**2, 1))
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

generator = Generator()
discriminator = Discriminator()
generator.cuda()
discriminator.cuda()
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
vis = visdom.Visdom(env='wgan')
dataloader = DataLoader(datasets.MNIST('./images', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5), (0.5))
                                        ])),
                                        batch_size=batch_size, shuffle=True, num_workers=n_cpu)

d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=lr)
g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=lr)

Tensor = torch.cuda.FloatTensor

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        batch_size = imgs.size(0)
        real_imgs = Variable(imgs.type(Tensor))

        d_optimizer.zero_grad()
        noise = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim))))

        fake_imgs = generator(noise).detach()
        d_loss = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
        d_loss.backward()
        d_optimizer.step()

        for p in discriminator.parameters():
            p.data.clamp_(-clamp, clamp)

        #g_optimizer.zero_grad()
        if i % n_critic == 0:
            g_optimizer.zero_grad()
            fake_imgs = generator(noise)
            g_loss = -torch.mean(discriminator(fake_imgs))
            g_loss.backward()
            g_optimizer.step()
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, epochs, i, len(dataloader), -d_loss.item(), -g_loss.item()))
    vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([-d_loss.item()]), win='d loss', update='append' if i> 0 else None)
    vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([-g_loss.item()]), win='g loss', update='append' if i> 0 else None)
    vis.images(fake_imgs, win='wgan')