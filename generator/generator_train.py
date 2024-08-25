import torch
from torch import nn
import os
from path import Paths
from generator_vae import RandomCharacterDataset, DataLoader
from generator.basic_gen import Generator, Discriminator

def train_gan(generator, discriminator, data_loader, device, num_epochs, latent_dim=500):
    criterion = nn.BCELoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(num_epochs):
        for real_chars in data_loader:
            real_chars = real_chars.to(device)
            batch_size = real_chars.size(0)

            # Train Discriminator
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_chars = generator(z)
            discriminator.zero_grad()
            real_loss = criterion(discriminator(real_chars), torch.ones(batch_size, 1, device=device))
            fake_loss = criterion(discriminator(fake_chars.detach()), torch.zeros(batch_size, 1, device=device))
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            generator.zero_grad()
            g_loss = criterion(discriminator(fake_chars), torch.ones(batch_size, 1, device=device))
            g_loss.backward()
            g_optimizer.step()
        print(f'Epoch {epoch + 1}, g_Loss: {g_loss.item()}, d_Loss: {d_loss.item()}')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Dataset and DataLoader setup
    dataset = RandomCharacterDataset(1000, 10, 10)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # training model
    output_dim = 500
    input_dim = 500
    gen = Generator(input_dim, output_dim).to(device)
    dis = Discriminator(input_dim, output_dim).to(device)
    train_gan(gen, dis, dataloader, device, num_epochs=10000)

    # save model
    path = Paths()
    model_save = os.path.join(path.TRAINED_MODEL, 'generator_gan.pth')