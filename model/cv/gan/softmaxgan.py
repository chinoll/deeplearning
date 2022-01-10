import numpy as np
from torch.autograd import Variable

import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self,latent_dim,img_size,channels):
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
    def __init__(self,channels,img_size):
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
        ds_size = img_size//2**4
        self.adv_layer = nn.Sequential(nn.Linear(128*ds_size**2, 1))
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

def model_init(latent_dim,img_size,channels,lr,b1,b2,weights_init,**kwargs):
    generator = Generator(latent_dim,img_size,channels)
    discriminator = Discriminator(channels,img_size)

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
Tensor = torch.cuda.FloatTensor
def train(epoch,dataloader,generator,discriminator,d_optimizer,g_optimizer,batch_size,latent_dim,vis,**kwargs):
    g_target = 1 / (batch_size * 2)
    d_target = 1 / batch_size
    for i, (imgs,_) in enumerate(dataloader):
        real_imgs = Variable(imgs.type(Tensor))

        g_optimizer.zero_grad()
        noise = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
        gen_imgs = generator(noise)

        # Train discriminator
        d_optimizer.zero_grad()
        pred_real = discriminator(real_imgs)
        pred_fake = discriminator(gen_imgs.detach())
        z = torch.sum(torch.exp(-pred_real)) + torch.sum(torch.exp(-pred_fake))
        d_loss = d_target * torch.sum(pred_real) + torch.log(z)
        d_loss.backward()
        d_optimizer.step()

        # Train generator
        gen_imgs = generator(noise)
        pred_fake = discriminator(gen_imgs)

        g_loss =  g_target * (torch.sum(pred_real.detach()) + torch.sum(pred_fake)) + torch.log(z.detach())
        g_loss.backward()
        g_optimizer.step()

    vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([g_loss.item()]), win='g loss', update='append' if i> 0 else None)
    vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([d_loss.item()]), win='d loss', update='append' if i> 0 else None)
    vis.images(gen_imgs, win='gen', opts=dict(title='Generated Images'))