import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np

class Generator(nn.Module):
    def __init__(self,latent_dim,channels,img_size,n_classes):
        super(Generator, self).__init__()
        self.emb = nn.Embedding(n_classes, latent_dim)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z,label):
        gen_labels = torch.mul(self.emb(label),z)
        out = self.l1(gen_labels)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv(out)
        return img

class Discriminator(nn.Module):
    def __init__(self,channels,img_size,n_classes):
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

        ds_size = img_size//2**4
        self.adv_layer = nn.Sequential(nn.Linear(128*ds_size**2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128*ds_size**2, n_classes), nn.Softmax(dim=0))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        return validity,label


def model_init(latent_dim,channels,img_size,n_classes,weights_init,lr,b1,b2,**kwargs):
    adv_loss = torch.nn.BCEWithLogitsLoss()
    aux_loss = torch.nn.CrossEntropyLoss()

    generator = Generator(latent_dim,channels,img_size,n_classes)
    discriminator = Discriminator(channels,img_size,n_classes)

    generator.cuda()
    discriminator.cuda()
    adv_loss.cuda()
    aux_loss.cuda()

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    return {
        "generator":generator,
        "discriminator":discriminator,
        "adv_loss":adv_loss,
        "aux_loss":aux_loss,
        "g_optimizer":g_optimizer,
        "d_optimizer":d_optimizer
    }

def train(epoch,generator,discriminator,dataloader,g_optimizer,d_optimizer,adv_loss,aux_loss,latent_dim,n_classes,vis,**kwargs):
    gloss_list = []
    dloss_list = []
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.shape[0]
        valid = Variable(torch.ones(batch_size, 1)).cuda()
        fake = Variable(torch.zeros(batch_size, 1)).cuda()

        real_imgs = Variable(imgs.type(torch.FloatTensor)).cuda()
        labels = Variable(labels.type(torch.LongTensor)).cuda()

        #训练生成器
        g_optimizer.zero_grad()

        z = Variable(torch.randn(batch_size, latent_dim)).cuda()
        gen_labels = Variable(torch.randint(0, n_classes, (batch_size,))).cuda()
        gen_imgs = generator(z,gen_labels)

        validity, pred_label = discriminator(gen_imgs)
        g_loss = (adv_loss(validity, valid) + aux_loss(pred_label, labels)) / 2
        gloss_list.append(g_loss.item())

        g_loss.backward()
        g_optimizer.step()

        #训练判别器
        d_optimizer.zero_grad()

        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (adv_loss(real_pred, valid) + aux_loss(real_aux, labels)) / 2

        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adv_loss(fake_pred, fake) + aux_loss(fake_aux, gen_labels)) / 2

        d_loss = (d_real_loss + d_fake_loss) / 2
        dloss_list.append(d_loss.item())

        d_loss.backward()
        d_optimizer.step()
    vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([np.mean(gloss_list)]), win='g loss', update='append' if i> 0 else None)
    vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([np.mean(dloss_list)]), win='d loss', update='append' if i> 0 else None)
    vis.images(gen_imgs, win='gen', opts=dict(title='Generated Images'))
    vis.images(real_imgs.detach(), win='real', opts=dict(title='Real Images'))