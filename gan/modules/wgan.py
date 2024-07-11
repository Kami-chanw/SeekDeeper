import torch
import torch.nn as nn
from .gan_base import GANBase


# Totally same as generator of DCGAN
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

        output_dim = n_channel
        for h_dim, K, S, P in zip(
            disc_hidden_dims,
            kernel_size[:-1],
            stride[:-1],
            padding[:-1],
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
        # The output of D is no longer a probability,
        # we do not apply sigmoid at the output of D.
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


class WGAN(GANBase):
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

        super(WGAN, self).__init__(
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

    def grad_penalty(self, real_samples, fake_samples):
        device = real_samples.device
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        out = self.discriminator(interpolates)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=out,
            inputs=interpolates,
            grad_outputs=torch.ones(out.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
