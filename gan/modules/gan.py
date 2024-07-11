import torch
import torch.nn as nn
from .gan_base import GANBase


class Generator(nn.Module):
    def __init__(self, z_dim, gen_hidden_dims, output_dim, device):
        super(Generator, self).__init__()
        layers = []
        input_dim = z_dim
        for h_dim in gen_hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim, device=device))
            layers.append(nn.BatchNorm1d(h_dim, device=device))
            layers.append(nn.ReLU(inplace=True))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, output_dim, device=device))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim, disc_hidden_dims, leaky_relu_slope, device):
        super(Discriminator, self).__init__()
        layers = []
        for h_dim in disc_hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim, device=device))
            layers.append(nn.LeakyReLU(leaky_relu_slope, inplace=True))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, 1, device=device))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).view(-1)


class GAN(GANBase):
    """
    A Generative Adversarial Network (GAN) class.

    Attributes:
        z_dim (int): Dimension of the noise vector (latent space).
        output_dim (int): Dimension of the generated output (e.g., image size).
        gen_hidden_dims (list of int): List of hidden layer dimensions for the generator.
        disc_hidden_dims (list of int): List of hidden layer dimensions for the discriminator.
        leaky_relu_slope (float): Negative slope for the LeakyReLU activation function.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
    """
    def __init__(
        self,
        z_dim,
        output_dim,
        gen_hidden_dims,
        disc_hidden_dims,
        leaky_relu_slope,
        device,
    ):
        super(GAN, self).__init__(
            Generator(z_dim, gen_hidden_dims, output_dim, device=device),
            Discriminator(
                output_dim, disc_hidden_dims, leaky_relu_slope, device=device
            ), 
        )
