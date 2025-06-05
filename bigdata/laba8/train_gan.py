# train_gan.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from generator import get_generator
from discriminator import get_discriminator

# Конфигурация
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
batch_size = 64
epochs = 50
lr = 0.0002
beta1 = 0.5

# Создание директории для сохранения изображений
os.makedirs("dcgan_results", exist_ok=True)

# Данные MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Нормализация в [-1, 1]
])
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Инициализация моделей
G = get_generator(latent_dim).to(device)
D = get_discriminator().to(device)

# Оптимизаторы
optimizer_G = optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.00005, betas=(0.5, 0.999))

# Функция потерь
criterion = nn.BCELoss()

# Фиксированный шум для визуализации
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)


def show_generated(generator, discriminator, epoch, loss_D, loss_G, real_imgs=None, save_path=None):
    # Генерируем изображения
    z = torch.randn(64, latent_dim, 1, 1).to(device)
    with torch.no_grad():
        fake_images = generator(z).cpu()
        # Получаем предсказания дискриминатора
        D_real_pred = discriminator(real_imgs[:64].to(device)).cpu().mean().item() if real_imgs is not None else None
        D_fake_pred = discriminator(fake_images.to(device)).cpu().mean().item()

    # Создаем фигуру
    fig = plt.figure(figsize=(12, 12))

    # Добавляем основное изображение
    grid = make_grid(fake_images, nrow=8, normalize=True)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(grid.permute(1, 2, 0))
    title = f"Epoch {epoch} | Loss_D: {loss_D:.4f} | Loss_G: {loss_G:.4f}"
    if D_real_pred is not None:
        title += f"\nD(real): {D_real_pred:.2f} | D(fake): {D_fake_pred:.2f}"
    ax1.set_title(title)
    ax1.axis('off')

    # Добавляем гистограмму предсказаний дискриминатора
    if real_imgs is not None:
        ax2 = fig.add_subplot(2, 1, 2)
        with torch.no_grad():
            real_preds = discriminator(real_imgs[:1000].to(device)).cpu().numpy()
            fake_preds = discriminator(generator(torch.randn(1000, latent_dim, 1, 1).to(device))).cpu().numpy()

        ax2.hist(real_preds, bins=50, alpha=0.5, label='Real', color='green')
        ax2.hist(fake_preds, bins=50, alpha=0.5, label='Fake', color='red')
        ax2.set_title("Discriminator Predictions Distribution")
        ax2.set_xlabel("Prediction (0=fake, 1=real)")
        ax2.set_ylabel("Count")
        ax2.legend()

    plt.tight_layout()

    # Сохраняем полную визуализацию
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.show()


# Обучение
for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(train_loader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # Метки
        real_labels = torch.full((batch_size, 1), 0.9, device=device)
        fake_labels = torch.full((batch_size, 1), 0.0, device=device)
        # ---- Обучение дискриминатора ----
        optimizer_D.zero_grad()

        # Ошибка на реальных изображениях
        real_output = D(real_imgs)
        loss_D_real = criterion(real_output, real_labels)

        # Ошибка на фейковых изображениях
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_imgs = G(noise)
        fake_output = D(fake_imgs.detach())
        loss_D_fake = criterion(fake_output, fake_labels)

        # Общая ошибка и обновление весов
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optimizer_D.step()

        # ---- Обучение генератора ----
        optimizer_G.zero_grad()
        fake_output = D(fake_imgs)
        loss_G = criterion(fake_output, real_labels)  # Обманываем D
        loss_G.backward()
        optimizer_G.step()

        # Логирование
        if i % 200 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(train_loader)}] "
                  f"Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")

    # Визуализация и сохранение
    if epoch % 5 == 0:
        with torch.no_grad():
            fake = G(fixed_noise).detach().cpu()
        # Сохраняем только сгенерированные изображения
        save_image(fake, f"dcgan_results/epoch_{epoch}_generated.png", nrow=8, normalize=True)
        # Сохраняем полный отчет с гистограммой
        show_generated(G, D, epoch, loss_D.item(), loss_G.item(), real_imgs,
                       save_path=f"dcgan_results/epoch_{epoch}_report.png")

# Сохранение моделей
torch.save(G.state_dict(), "dcgan_generator.pth")
torch.save(D.state_dict(), "dcgan_discriminator.pth")