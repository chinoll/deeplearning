import numpy as np
from torch.autograd import Variable

import torch.nn as nn
import torch
class Generator(nn.Module):
    def __init__(self,img_size,latent_dim,channels):
        super(Generator, self).__init__()
        self.size = img_size//4
        self.linear = nn.Linear(latent_dim, 128*self.size**2)

        self.conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
    def forward(self, z):
        out = self.linear(z)
        out = out.view(out.shape[0], 128, self.size, self.size)
        img = self.conv(out)
        return img

class Discriminator(nn.Module):
    def __init__(self,img_size,channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
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
        down_dim = 64 * (img_size // 2) ** 2
        ds_size = img_size//2**4
        self.down_size = img_size // 2
        self.adv_layer = nn.Sequential(nn.Linear(128*ds_size**2, down_dim), nn.LeakyReLU(0.2, inplace=True))
        self.up = nn.Sequential(nn.Upsample(scale_factor=2))
        self.conv = nn.Sequential(nn.Conv2d(64, channels,3, stride=1,padding=1))
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        out = self.adv_layer(out).view(-1,64,self.down_size,self.down_size)
        out = self.up(out)
        out = self.conv(out)
        return out




Tensor = torch.cuda.FloatTensor
def model_init(latent_dim,img_size,channels,b1,b2,lr,weights_init,**kwargs):
    generator = Generator(img_size,latent_dim,channels)
    discriminator = Discriminator(img_size,channels)
    generator.cuda()
    discriminator.cuda()

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    return {
        "generator":generator,
        "discriminator":discriminator,
        "g_optimizer":g_optimizer,
        "d_optimizer":d_optimizer
    }
k = 0.0

def train(epoch,dataloader,generator,discriminator,g_optimizer,d_optimizer,vis,latent_dim,gamma,lambda_k,**kwargs):
    global k
    for i, (imgs,_) in enumerate(dataloader):
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(imgs.type(Tensor))

        g_optimizer.zero_grad()
        noise = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
        gen_imgs = generator(noise)

        # Train discriminator
        d_optimizer.zero_grad()
        d_real = discriminator(real_imgs)
        d_fake = discriminator(gen_imgs.detach())

        d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
        d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
        d_loss = d_loss_real - k * d_loss_fake

        d_loss.backward()
        d_optimizer.step()

        # Train generator
        gen_imgs = generator(noise)
        g_loss = torch.mean(torch.abs(discriminator(gen_imgs) - gen_imgs))
        g_loss.backward()
        g_optimizer.step()

        diff = torch.mean(gamma * d_loss_real - d_loss_fake)
        k += lambda_k * diff.item()
        k = min(max(k, 0), 1)
        M = (d_loss_real - torch.abs(diff)).item()

    vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([g_loss.item()]), win='g loss', update='append' if i> 0 else None)
    vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([d_loss.item()]), win='d loss', update='append' if i> 0 else None)
    vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([M]), win='M', update='append' if i> 0 else None)
    vis.images(gen_imgs, win='gen', opts=dict(title='Generated Images'))