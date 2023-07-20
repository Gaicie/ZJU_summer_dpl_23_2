import torch
import torch.nn as nn
import numpy as np

# 先计算好 time embedded 后面直接拿来用


# 本版本还是使用 gan 原来的 generator
# 下一版本再换成 diffusion 的 decoder
class Generator(nn.Module):
    def __init__(self,latent_dim,img_shape):
        super(Generator, self).__init__()
        self.img_shape=img_shape
        def block(in_feat, out_feat, normalize=True):
            # 每个 block 首先包含一个线性变换(y=Ax+b)，输入是一个维度为 in_feat 的向量，输出是一个维度为 out_feat 的向量
            layers = [nn.Linear(in_feat, out_feat)]
            # 如果需要 normalize 的话，就在线性变换之后再加一个正则化
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            # 最后是用 leaky relu 做激活函数
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # 初始输入除了噪声向量之外，再加上一个位置给 time step，即输入维度为 785 维
            *block(latent_dim + 1, 1024, normalize=False),
            *block(1024, 2048),
            *block(2048, 4096),
            *block(4096, 4096),
            nn.Linear(4096, int(np.prod(img_shape))),
            nn.Tanh()
        )

    '''
    input z: 一些噪声向量（维度为 opt.latent_dim）组成的矩阵
    input t: 当前迭代到第几步 time step
    '''

    def forward(self, z, t):
        # z 是一个 (z.size[0] * opt.latent_dim) 的矩阵
        # 先用 t 填充一个 (z.size[0] * 1) 的矩阵 torch.full((z.size(0), 1), t)
        # 然后拼接在一起构成 generator 的输入矩阵 (z.size[0] * (opt.latent_dim+1))
        z = z.to(device = "cpu")
        t = torch.full((z.size(0), 1), t).to(device = "cpu")
        z = torch.cat((z, t), dim=1).to(device = "cpu")

        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
        # 加上 t 转 time embedded 的函数