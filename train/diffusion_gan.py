import argparse
import os
import numpy as np
import math

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from typing import Dict, Tuple

from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from models.generator import Generator
from models.discriminator import Discriminator

os.makedirs("result/diffusion_gan_images", exist_ok=True)
os.makedirs('result/diffusion_gan_models', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=28*28, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
# 再增加一个超参数 time_step，表示 diffusion 的迭代次数
parser.add_argument("--time_step", type=int, default=100, help="diffusion iteration time")
# 再增加一个超参数 time_step_interval，表示间隔多少 time step 输出一个图片看看
parser.add_argument("--time_step_interval", type=int, default=100, help="no description")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False




def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 7, padding=3),
    nn.BatchNorm2d(oc),
    nn.LeakyReLU(),
)

class DummyEpsModel(nn.Module):
    def __init__(self, n_channel: int) -> None:
        super(DummyEpsModel, self).__init__()
        self.conv = nn.Sequential(  # with batchnorm
            blk(n_channel, 64),
            blk(64, 128),
            blk(128, 256),
            blk(256, 512),
            blk(512, 256),
            blk(256, 128),
            blk(128, 64),
            nn.Conv2d(64, n_channel, 3, padding=1),
        )

    def forward(self, x, t) -> torch.Tensor:
        # Lets think about using t later. In the paper, they used Tr-like positional embeddings.
        return self.conv(x)


class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
    ) -> None:
        super(DDPM, self).__init__()
        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T

    def add_noise(self, x, y, t):
        sqrtab_t = self.sqrtab[t]
        sqrtmab_t = self.sqrtmab[t]
        eps = torch.randn_like(x)  # eps ~ N(0, 1)
        x_noisy = sqrtab_t * x + sqrtmab_t * eps
        y_noisy = sqrtab_t * y + sqrtmab_t * eps
        return x_noisy, y_noisy

# Initialize generator and discriminator
generator = Generator(opt.latent_dim,img_shape)
discriminator = Discriminator(img_shape)

# Loss function
adversarial_loss = torch.nn.BCELoss()

# DDPM module to add noise
ddpm = DDPM(eps_model=DummyEpsModel(1), betas=(opt.b1, opt.b2), n_T=opt.time_step)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    ddpm.cuda()

# Configure data loader
os.makedirs("./data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
eta = np.linspace(0.0, 1.0 / opt.time_step, num=opt.time_step)

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        batches_done = epoch * len(dataloader) + i
        optimizer_G.zero_grad()
        
        # 初始化两个 loss 为 0
        g_loss = adversarial_loss(valid, valid)
        d_loss = adversarial_loss(valid, valid)

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        # Generate a batch of images
        gen = generator(z)
        for t in range(opt.time_step):
            r, g= ddpm.add_noise(real_imgs, gen, t)
            g_loss += eta[t] * adversarial_loss(discriminator(r), discriminator(g))

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        gen = generator(z)
        for t in range(opt.time_step):
            r, g= ddpm.add_noise(real_imgs, gen, t)
            d_loss += eta[t] * (adversarial_loss(discriminator(r), valid) + adversarial_loss(discriminator(g), fake)) / 2
            if batches_done % opt.sample_interval == 0 and (t+1) % opt.time_step_interval == 0:
                save_image(gen.data[:25], "result/diffusion_gan_images/%d_%i.png" % (batches_done, t+1), nrow=5, normalize=True)

        d_loss.backward()
        optimizer_D.step()
        if batches_done % opt.sample_interval == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

    torch.save(generator.state_dict(), "result/diffusion_gan_models/generator_%d.pth" % epoch)
    torch.save(discriminator.state_dict(), "result/diffusion_gan_models/discriminator_%d.pth" % epoch)
