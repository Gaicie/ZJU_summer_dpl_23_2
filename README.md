# ZJU_summer_dpl_23_2

## DifGAN PyTorch Implementation

This is the repository of the course project implement of a summer course called "Cutting-edge Technologies and Applications of Deep Learning" taught by Hehe Fan in ZJU.
We gave four methods in `./train`, they are original GAN, high dimension GAN, reimplementation of diffusion GAN in [Diffusion-GAN: Training GANs with Diffusion](https://arxiv.org/abs/2206.02262), and our own DifGAN respectively.

`./models` includes the models of discriminator and generator (with and without tilmestep as input).

`./result` should record the generated images and the model parameters(`.pth`) after training.

There is no `./dataset` folder as we use Mnist in official library.

### Train DifGAN

* 64-bit Python 3.7 or newer version
* PyTorch 1.7.1 or newer version. See https://pytorch.org/ for PyTorch install instructions.
* CUDA or no CUDA are both available (only CPU would be slow though)

#### Data Preparation

In the project we trained our model on [CIFAR (32 x 32)](https://www.cs.toronto.edu/~kriz/cifar.html) and [MNIST (28 x 28)](http://yann.lecun.com/exdb/mnist/)
It is not necessary to download the dataset separately, the downloading is embedded in current code

#### Training

e.g. If you want to train gan model, you can run `python -m train.gan` under project directory
