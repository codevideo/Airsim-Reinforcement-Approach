import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

# Set GPU
torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set folder
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Set Hyper parameters
image_size = 784
h_dim = 400
z_dim = 20
num_epochs = 15
batch_size = 128
learning_rate = 1e-3

# Got data & Data loader
dataset = torchvision.datasets.MNIST(root='./minist',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=False)
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

# VAE model
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)        # Input layer
        self.fc2 = nn.Linear(h_dim, z_dim)             # Hidden layer(mean)
        self.fc3 = nn.Linear(h_dim, z_dim)             # Hidden layer(var)
        # latent space
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

    # Encoder
    def encode(self, img):
        h = F.relu(self.fc1(img))
        mean = self.fc2(h)
        var = self.fc3(h)

        return mean, var

    # mix m, var, noise
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)

        return mu + eps * std

    # Decoder
    def decode(self, z):
        h = F.relu(self.fc4(z))
        out_img = F.sigmoid(self.fc5(h))

        return out_img

    # forward
    def forward(self, img):
        m, var = self.encode(img)
        z = self.reparameterize(m, var)
        img_reconst = self.decode(z)

        return img_reconst, m, var

# Define the model & Optimizer
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (x, _) in enumerate(data_loader):
        # Get data
        #print(x.shape) $torch.Size([128, 1, 28, 28])
        x = x.to(device).view(-1, image_size)
        x_reconst, mu, log_var = model(x)

        # Loss
        reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Backward
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item(), kl_div.item()))

    # Testing model
    with torch.no_grad():
        # Random sample img
        z = torch.randn(batch_size, z_dim).to(device)
        out = model.decode(z).view(-1, 1, 28, 28)
        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)))

        # Reconst Img
        out, _, _ = model(x)
        x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))
