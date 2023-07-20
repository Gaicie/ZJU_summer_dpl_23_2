import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self,latent_dim,img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # generator 模型由一些 block 组成
        # 先把输入向量层层放大到 1024 维，最后再缩小到图片大小
        # 最后再加一个 tanh 激活函数，把结果放缩到 0 和 1 之间，然后取整变成 0 或者 1
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    '''
    input z: 一些噪声向量（维度为 opt.latent_dim）组成的矩阵
    '''
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
