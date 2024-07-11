import torch
import os

torch.manual_seed(3407)

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model parameter setting
batch_size = 128
z_dim = 100  # dimension for noise
leaky_relu_slope = 0.2


# training parameter setting
gen_lr = 2e-4
disc_lr = 2e-4
epochs = 30
n_critic = 5  # train generator for every n_critic steps, only used in WGANs.
lambda_gp = 10  # for WGAN-GP

betas = (0.5, 0.999)  # beta for Adam
betas_gp = (0.0, 0.9)  # betas for Adam of WGAN-GP

# sampling parameter setting
sample_interval = 3  # Interval of epochs to sample and save generated images
num_samples = 64  # Number of images to generate each sampling


# path
ckpt_mnist_dir = "./checkpoints/mnist"
sample_mnist_dir = "./samples/mnist"

ckpt_cifar10_dir = "./checkpoints/cifar10"
sample_cifar10_dir = "./samples/cifar10"

os.makedirs(ckpt_mnist_dir, exist_ok=True)
os.makedirs(ckpt_cifar10_dir, exist_ok=True)
