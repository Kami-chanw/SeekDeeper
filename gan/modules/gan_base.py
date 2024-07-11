import torch


class GANBase:
    def __init__(
        self,
        generator,
        discriminator,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.expected_input_size = None

    def generate(self, z):
        return self.generator(z)

    def discriminate(self, images):
        return self.discriminator(images)

    def state_dict(self):
        return {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.generator.load_state_dict(state_dict["generator"])
        self.discriminator.load_state_dict(state_dict["discriminator"])

    def train(self):
        self.generator.train()
        self.discriminator.train()
    
    def eval(self):
        self.generator.eval()
        self.discriminator.eval()
