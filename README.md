# ZJU_summer_dpl_23_2
This is the repository of the course project implement of a summer course called " Deep Learning" taught by Hehe Fan in ZJU.
We implement four methods in `./train` , original GAN, high dimension GAN, diffusion GAN in [Diffusion-GAN: Training GANs with Diffusion](https://arxiv.org/abs/2206.02262), and our own DifGAN.

`./models` includes the models of discriminator and generator (with and without tilmestep as input).

`./result` should record the generated images and the model parameters(`.pth`) after training.

There is no dataset because we use Mnist in official library.
