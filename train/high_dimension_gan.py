import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from models.generator import Generator
from models.discriminator import Discriminator
os.makedirs("result/high_d_images", exist_ok=True)
os.makedirs("result/high_d_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# 把输入噪声的维度提高到跟原图一样，测试一下生成的结果
parser.add_argument("--latent_dim", type=int, default= 28 * 28, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=8000, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False




# Loss function
# binary cross entropy 一个针对二分类问题的损失函数
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator(opt.latent_dim,img_shape)
discriminator = Discriminator(img_shape)

# 如果要导入已经训练好的模型，就加上下面这两句
# generator.load_state_dict("generator.pth")
# discriminator.load_state_dict("discriminator.pth")

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

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

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        # 维度为 imgs 的大小，全 1 和全 0 的两个向量
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        # 数据集中取出来的真实世界图像，转换成向量
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        # G 所有梯度归零
        optimizer_G.zero_grad()

        # Sample noise as generator input
        # 用标准正态分布随机一个 (imgs 中图片数量) * (预定义的噪声向量长度) 的噪声矩阵
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        # 这里是用 D 判断生成的图片是否为真实世界图像，结果和全 1 的序列相比
        # G 越能够骗过 D，判断结果中 1 就越多，最终 loss 就越小
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        # 反向传播，训练 G
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # D 的梯度先清零
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        # 反向传播，训练 D
        d_loss.backward()
        optimizer_D.step()

        batches_done = epoch * len(dataloader) + i

        # 在训练过程中，输出一些 G 生成的图片检查一下
        # 减少了中间输出和图片的输出，加快训练过程（大概可以）
        if batches_done % opt.sample_interval == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
            save_image(gen_imgs.data[:25], "result/high_d_images/%d.png" % batches_done, nrow=5, normalize=True)

    torch.save(generator.state_dict(), "result/high_d_models/generator_%d.pth" % epoch)
    torch.save(discriminator.state_dict(), "result/high_d_models/discriminator_%d.pth" % epoch)

# ----------
#  Generating
# ----------

