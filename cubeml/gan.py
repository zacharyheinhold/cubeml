import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GAN:
    def __init__(self, input_dim, predictor, lr=0.001, batch_size=100, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.input_dim = input_dim
        self.predictor = predictor.to(device).eval()
        self.device = device

        self.generator = self.Generator(input_dim).to(self.device)
        self.discriminator = self.Discriminator(input_dim).to(self.device)

        # Set up optimizers here
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr)

        self.criterion = nn.BCELoss()

        self.batch_size = batch_size

    class Generator(nn.Module):
        def __init__(self, input_dim, d_model=64, num_dense_layers=3):
            super(GAN.Generator, self).__init__()
            
            self.denses = nn.ModuleList([nn.Linear(d_model // (2 ** i), d_model // (2 ** (i + 1))) for i in range(num_dense_layers)])
            self.output = nn.Linear(d_model // (2 ** num_dense_layers), input_dim)

        def forward(self, z):
            x = z
            for dense in self.denses:
                x = nn.ReLU()(dense(x))
            x = self.output(x)
            return x

    class Discriminator(nn.Module):
        def __init__(self, input_dim, d_model=64, num_dense_layers=3):
            super(GAN.Discriminator, self).__init__()
            
            self.denses = nn.ModuleList([nn.Linear(d_model // (2 ** i), d_model // (2 ** (i + 1))) for i in range(num_dense_layers)])
            self.output = nn.Linear(d_model // (2 ** num_dense_layers), 1)

        def forward(self, x):
            for dense in self.denses:
                x = nn.ReLU()(dense(x))
            x = self.output(x)
            return nn.Sigmoid()(x)

    def train(self, data, num_epochs):
        for epoch in range(num_epochs):
            try:
                for real_samples in data:
                    real_samples = real_samples.to(self.device)
                    batch_size = real_samples.size(0)
                    
                    # Labels for real and fake data
                    real_labels = torch.ones(batch_size, 1).to(self.device)
                    fake_labels = torch.zeros(batch_size, 1).to(self.device)

                    # Generate fake samples
                    noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                    fake_samples = self.generator(noise)

                    # Get logits for the fake samples using the predictor
                    logits = self.predictor(fake_samples)

                    # Train discriminator
                    self.disc_optimizer.zero_grad()
                    
                    outputs = self.discriminator(real_samples, logits)
                    d_loss_real = self.criterion(outputs, real_labels)
                    
                    outputs = self.discriminator(fake_samples.detach(), logits.detach())
                    d_loss_fake = self.criterion(outputs, fake_labels)
                    
                    d_loss = d_loss_real + d_loss_fake
                    d_loss.backward()
                    self.disc_optimizer.step()

                    # Train generator
                    self.gen_optimizer.zero_grad()
                    
                    outputs = self.discriminator(fake_samples, logits)
                    g_loss = self.criterion(outputs, real_labels)  # We want the discriminator to believe that fake samples are real
                    g_loss.backward()
                    self.gen_optimizer.step()
                
                print(f"Epoch [{epoch + 1}/{num_epochs}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

            except KeyboardInterrupt:
                print("Training interrupted by user. Saving current state.")
                break

        return fake_samples.cpu().detach()
