import torch.nn as nn
from .gan_base import GANBase


class Generator(nn.Module):
    def __init__(
        self, z_dim, n_channel, gen_hidden_dims, kernel_size, stride, padding, device
    ):
        super(Generator, self).__init__()
        layers = []
        input_dim, output_dim = z_dim, gen_hidden_dims[0]
        layers.append(
            nn.ConvTranspose2d(
                input_dim,
                output_dim,
                kernel_size=kernel_size[0],
                stride=stride[0],
                padding=padding[0],
                bias=False,
                device=device,
            )
        )
        for h_dim, K, S, P in zip(
            gen_hidden_dims[1:] + [n_channel], kernel_size[1:], stride[1:], padding[1:]
        ):
            layers.append(nn.BatchNorm2d(output_dim, device=device))
            layers.append(nn.ReLU(inplace=True))
            input_dim, output_dim = output_dim, h_dim
            layers.append(
                nn.ConvTranspose2d(
                    input_dim,
                    output_dim,
                    kernel_size=K,
                    stride=S,
                    padding=P,
                    bias=False,
                    device=device,
                )
            )
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(
        self,
        n_channel,
        disc_hidden_dims,
        kernel_size,
        stride,
        padding,
        leaky_relu_slope,
        device,
    ):
        super(Discriminator, self).__init__()
        layers = []
        input_dim, output_dim = n_channel, disc_hidden_dims[0]
        layers.append(
            nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size=kernel_size[0],
                stride=stride[0],
                padding=padding[0],
                bias=False,
                device=device,
            )
        )
        layers.append(nn.LeakyReLU(leaky_relu_slope, inplace=True))
        for h_dim, K, S, P in zip(
            disc_hidden_dims[1:],
            kernel_size[1:],
            stride[1:],
            padding[
                1:
            ],  # the last kernel_size, stride and padding will not be used here
        ):
            input_dim, output_dim = output_dim, h_dim
            layers.append(
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=K,
                    stride=S,
                    padding=P,
                    bias=False,
                    device=device,
                )
            )
            layers.append(nn.BatchNorm2d(output_dim, device=device))
            layers.append(nn.LeakyReLU(leaky_relu_slope, inplace=True))
        layers.append(
            nn.Conv2d(
                output_dim,
                1,
                kernel_size=kernel_size[-1],
                stride=stride[-1],
                padding=padding[-1],
                bias=False,
                device=device,
            )
        )
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

        output_size = 1
        for K, S, P in zip(reversed(kernel_size), reversed(stride), reversed(padding)):
            output_size = (output_size - 1) * S - 2 * P + K

        if output_size < 1:
            raise ValueError(
                "Wrong parameters for convolutional kernel size, stride or padding."
            )
        self.expected_input_size = output_size

    def forward(self, x):
        b, c, h, w = x.shape

        if w != h or w != self.expected_input_size:
            raise ValueError(
                f"Expect shape {c}x{self.expected_input_size}x{self.expected_input_size}, but got {c}x{h}x{w}."
            )

        return self.layers(x).view(-1)


class DCGAN(GANBase):
    """
    Deep Convolutional Generative Adversarial Network (DCGAN) class.

    Attributes:
        z_dim (int): Dimension of the noise vector (latent space).
        n_channel (int): Number of channels in the generated output (e.g., for RGB images, n_channel=3).
        gen_hidden_dims (list of int): List of hidden layer channels for the generator.
        gen_kernel_size (list of int): List of kernel sizes for each convolutional layer in the generator.
        gen_stride (list of int): List of strides for each convolutional layer in the generator.
        gen_padding (list of int): List of paddings for each convolutional layer in the generator.
        disc_hidden_dims (list of int): List of hidden layer channels for the discriminator.
        disc_kernel_size (list of int): List of kernel sizes for each convolutional layer in the discriminator.
        disc_stride (list of int): List of strides for each convolutional layer in the discriminator.
        disc_padding (list of int): List of paddings for each convolutional layer in the discriminator.
        leaky_relu_slope (float): Negative slope for the LeakyReLU activation function.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
    """
    def __init__(
        self,
        z_dim,
        n_channel,
        gen_hidden_dims,
        gen_kernel_size,
        gen_stride,
        gen_padding,
        disc_hidden_dims,
        disc_kernel_size,
        disc_stride,
        disc_padding,
        leaky_relu_slope,
        device,
    ):
        def check_params(hidden_dims, kernel_size, stride, padding):
            if (
                not len(kernel_size)
                == len(stride)
                == len(padding)
                == len(hidden_dims) + 1
            ):
                raise ValueError(
                    "Convolutional layer parameters are incorrect. Please make sure that the number of kernel_size, stride and padding is the same and one greater than the number of hidden_dims."
                )

        check_params(gen_hidden_dims, gen_kernel_size, gen_stride, gen_padding)
        check_params(disc_hidden_dims, disc_kernel_size, disc_stride, disc_padding)

        super(DCGAN, self).__init__(
            Generator(
                z_dim,
                n_channel,
                gen_hidden_dims,
                kernel_size=gen_kernel_size,
                stride=gen_stride,
                padding=gen_padding,
                device=device,
            ),
            Discriminator(
                n_channel,
                disc_hidden_dims,
                kernel_size=disc_kernel_size,
                stride=disc_stride,
                padding=disc_padding,
                leaky_relu_slope=leaky_relu_slope,
                device=device,
            ),
        )

        def weights_init(m): # suggested in DCGAN paper
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
