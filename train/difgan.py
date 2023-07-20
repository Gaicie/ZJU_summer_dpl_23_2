import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from pytorch_fid import fid_score
import torch.nn as nn
import torch.nn.functional as F
import torch
from models.generator_t import Generator
from models.discriminator import Discriminator

os.makedirs("result/difgan_images", exist_ok=True)
os.makedirs("result/difgan_models", exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# 把输入噪声的维度提高为图片大小
parser.add_argument("--latent_dim", type=int, default=28 * 28 * 1, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image samples")
# 再增加一个超参数 time_step，表示 diffusion 的迭代次数
parser.add_argument("--time_step", type=int, default=100, help="diffusion iteration time")
# 再增加一个超参数 time_step_interval，表示间隔多少 time step 输出一个图片看看
parser.add_argument("--time_step_interval", type=int, default=10, help="no description")
opt = parser.parse_args()
print(opt)

# 新增超参数 eta, 表示每个 time step 中 loss 的系数
eta = np.linspace(0.0, 1.0 / opt.time_step, num=opt.time_step)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


# 输入一个数字 timestep，输出一个 dim 维的向量
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding








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

        # Sample noise as generator input
        # 用标准正态分布随机一个 (imgs 中图片数量) * (预定义的噪声向量长度) 的噪声矩阵
        # 其中每个元素都是 float
        gen_imgs = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        batches_done = epoch * len(dataloader) + i

        # G 所有梯度归零
        optimizer_G.zero_grad()

        # D 的梯度先清零
        optimizer_D.zero_grad()

        # 初始化两个 loss 为 0
        g_loss = adversarial_loss(valid, valid)
        d_loss = adversarial_loss(valid, valid)

        for t in range(opt.time_step):

            # Generate a batch of images
            gen_imgs = gen_imgs.flatten(1)
            gen_imgs = generator(gen_imgs.data, t)  # 加上 .data 就不会报错了

            # Loss measures generator's ability to fool the discriminator
            # 这里是用 D 判断生成的图片是否为真实世界图像，结果和全 1 的序列相比
            # G 越能够骗过 D，判断结果中 1 就越多，最终 loss 就越小
            # loss 乘上一个系数 eta[t]
            g_loss += eta[t] * adversarial_loss(discriminator(gen_imgs), valid)

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss += eta[t] * (real_loss + fake_loss) / 2

            # 在训练过程中，输出一些 G 生成的图片检查一下
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "result/difgan_images/%d.png" % batches_done, nrow=5,
                       normalize=True)

        # 跳出 t 的循环，回到 i 的循环
        # 反向传播，训练 G
        g_loss.backward()
        optimizer_G.step()

        # 反向传播，训练 D
        d_loss.backward()
        optimizer_D.step()

        if batches_done % opt.sample_interval == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

    torch.save(generator.state_dict(), "result/difgan_models/generator.pth")
    torch.save(discriminator.state_dict(), "result/difgan_models/discriminator.pth")
