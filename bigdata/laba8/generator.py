import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            # Преобразуем входной вектор (latent_dim) в тензор 7x7x256
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Увеличиваем размер до 14x14x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Увеличиваем размер до 28x28x1
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Нормализуем выход в [-1, 1]
        )

    def forward(self, z):
        z = z.view(-1, self.latent_dim, 1, 1)
        return self.main(z)


def get_generator(latent_dim=100):
    return Generator(latent_dim)