import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from tqdm import trange
import time
import numpy as np

import utils

# Set GPU
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# Set folder
sample_dir = 'dronet_samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Got data & Data loader
img_buffer = utils.ImgBuffer(8, 120, 160)
img_buffer.load("/home/pc3396/newrtd3_test/NewRTD3/Weight/fullinfo_vel_maxv5_6/ImgBuffer/TD3_Soccer_Field_Easy_0_pretrained_fullinfo_maxV5.npz")

# Set Hyper parameters
image_high = img_buffer.img_high
image_width = img_buffer.img_width
image_size = image_high*image_width*3
h_dim_1 = 1000
h_dim_2 = 500
h_dim_3 = 250
z_dim = 10
num_epochs = 400
ep_max_step = 10000
batch_size = 32
learning_rate = 5e-4


INPUT_CHANNELS = 3
CHANNEL_FACTOR = 0.25
DROPOUT_PROBABILITY = 0.5
DATA_BATCH_SIZE = 10

# VAE model
class VAE(nn.Module):
    def __init__(self, image_size, h_dim_1, h_dim_2, h_dim_3, z_dim, input_channels=INPUT_CHANNELS, channel_factor=CHANNEL_FACTOR, dropout_probability=DROPOUT_PROBABILITY, batch_size=DATA_BATCH_SIZE):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim_1)        # Input layer
        self.fc2_1 = nn.Linear(h_dim_1, h_dim_2)
        self.fc2_2 = nn.Linear(h_dim_2, h_dim_3)
        self.fc3 = nn.Linear(h_dim_3, z_dim)             # Hidden layer(mean)
        self.fc4 = nn.Linear(h_dim_3, z_dim)             # Hidden layer(var)
        

        # DroNet Architecture
        # Layer 0
        self.layer_0_conv1 = nn.Conv2d(input_channels, int(32*channel_factor), 5, stride=2, padding=2)
        self.layer_0_maxpool1 = nn.MaxPool2d(3, stride=2)

        # Layer 1
        self.layer_1_1_conv1 = nn.Conv2d(int(32*channel_factor), int(32*channel_factor), 3, stride=2, padding=1)
        self.layer_1_1_conv2 = nn.Conv2d(int(32*channel_factor), int(32*channel_factor), 3, padding=1)
        self.layer_1_2_conv1 = nn.Conv2d(int(32*channel_factor), int(32*channel_factor), 1, stride=2)

        # Layer 2
        self.layer_2_1_conv1 = nn.Conv2d(int(32*channel_factor), int(64*channel_factor), 3, stride=2, padding=1)
        self.layer_2_1_conv2 = nn.Conv2d(int(64*channel_factor), int(64*channel_factor), 3, padding=1)
        self.layer_2_2_conv1 = nn.Conv2d(int(32*channel_factor), int(64*channel_factor), 1, stride=2)

        # Layer 3
        self.layer_3_1_conv1 = nn.Conv2d(int(64*channel_factor), int(128*channel_factor), 3, stride=2, padding=1)
        self.layer_3_1_conv2 = nn.Conv2d(int(128*channel_factor), int(128*channel_factor), 3, padding=1)
        self.layer_3_2_conv1 = nn.Conv2d(int(64*channel_factor), int(128*channel_factor), 1, stride=2)

        # Layer 4
        self.layer_4_dropout = nn.Dropout(dropout_probability)
        self.layer_4_linear = nn.Linear(4 * 5 * int(128*channel_factor), int(256*channel_factor))

        # Layer 5 & 6
        self.layer_5_linear = nn.Linear(int(256*channel_factor), z_dim)
        self.layer_6_linear = nn.Linear(int(256*channel_factor), z_dim)
        
        # latent space
        self.fc5 = nn.Linear(z_dim, h_dim_3)
        self.fc6_1 = nn.Linear(h_dim_3, h_dim_2)
        self.fc6_2 = nn.Linear(h_dim_2, h_dim_1)
        self.fc7 = nn.Linear(h_dim_1, image_size)

    # Encoder
    def encode(self, img):
        # h = F.relu(self.fc1(img))
        # h = F.relu(self.fc2_1(h))
        # h = F.relu(self.fc2_2(h))
        # mean = self.fc3(h)
        # var = self.fc4(h)

        # DroNet architecture
        # Layer 0
        x0 = self.layer_0_conv1(img)
        x0 = self.layer_0_maxpool1(x0)

        ##########
        # Layer 1
        ##########
        # Layer 1_1
        x11 = F.relu(x0)
        x11 = self.layer_1_1_conv1(x11)
        x11 = F.relu(x11)
        x11 = self.layer_1_1_conv2(x11)

        # Layer 1_2
        x12 = self.layer_1_2_conv1(x0)

        # Layer 1 Total
        x11.add(x12)
        x1 = x11

        ##########
        # Layer 2
        ##########
        # Layer 2_1
        x21 = F.relu(x1)
        x21 = self.layer_2_1_conv1(x21)
        x21 = F.relu(x21)
        x21 = self.layer_2_1_conv2(x21)

        # Layer 2_2
        x22 = self.layer_2_2_conv1(x1)

        # Layer 2 Total
        x21.add(x22)
        x2 = x21

        ##########
        # Layer 3
        ##########
        # Layer 3_1
        x31 = F.relu(x2)
        x31 = self.layer_3_1_conv1(x31)
        x31 = F.relu(x31)
        x31 = self.layer_3_1_conv2(x31)

        # Layer 2_2
        x32 = self.layer_3_2_conv1(x2)

        # Layer 2 Total
        x31.add(x32)
        x3 = x31

        ##########
        # Layer 4
        ##########
        x4 = torch.reshape(x3, (x3.shape[0], -1,))
        
        x4 = self.layer_4_dropout(x4)
        #print('x4.shape',x4.shape)
        x4 = self.layer_4_linear(x4)

        ##########
        # Layer 5
        ##########
        mean = self.layer_5_linear(F.relu(x4))
        var = self.layer_6_linear(F.relu(x4))

        # return x5

        return mean, var

    # mix m, var, noise
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)

        return mu + eps * std

    # Decoder
    def decode(self, z):
        h = F.relu(self.fc5(z))
        h = F.relu(self.fc6_1(h))
        h = F.relu(self.fc6_2(h))
        out_img = F.sigmoid(self.fc7(h))

        return out_img

    # forward
    def forward(self, img):
        m, var = self.encode(img)
        z = self.reparameterize(m, var)
        img_reconst = self.decode(z)

        return img_reconst, m, var


# Define the model & Optimizer
model = VAE(image_size, h_dim_1, h_dim_2, h_dim_3, z_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epoch_loss = []
t_range = trange(num_epochs,desc='loss pred & time',leave=True)
#for epoch in tqdm(range(num_epochs)):
'''
print('test save image')
_, x = img_buffer.sample(32)

print('init x.shape',x.shape)
x = x[:,:,:,:,[-1,-2,-3]] # x.shape = batch * 1 * height * width * 3

print('shift x.shape',x.shape)

x = torch.squeeze(torch.transpose(torch.transpose(x,-1,-2),-2,-3)) 
print('shift3 x.shape',x.shape)
save_image(x/255,'img.png')
print('x.shape',x[0][0].permute(-1,-3,-2).shape)
stop()
'''


for epoch in t_range:
    for i in range(ep_max_step):
        # Get data
        sate, x = img_buffer.sample(batch_size)
        x_img = torch.squeeze(x/255).permute(-4,-1,-3,-2)
        x = x.view(-1, image_size)/255
        #print('x_img.shape:',x_img.shape)
        # x_reconst, mu, log_var = model(x)
        x_reconst, mu, log_var = model(x_img)

        # Loss
        # reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
        reconst_loss = F.mse_loss(x_reconst, x, size_average=False)
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Backward
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            #print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
            #       .format(epoch+1, num_epochs, i+1, ep_max_step, reconst_loss.item(), kl_div.item()))
            loss_string = ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, ep_max_step, reconst_loss.item(), kl_div.item()))
            t_range.set_description(loss_string)
    epoch_loss.append(reconst_loss.item())
    if (int(epoch) + 1) % 5 == 0: 
        torch.save(model.state_dict(), "dronet_model_folder/epoch"+str(epoch)+".pth")
        np.save('dronet_loss_folder/epoch_loss',epoch_loss)

    # Testing model
    with torch.no_grad():
        # Random sample img
        z = torch.randn(batch_size, z_dim).to(device)
#        out = model.decode(z).view(-1, 1, image_high, image_width)
        out = model.decode(z).view(-1, 1, image_high, image_width, 3)[:,:,:,:,[-1,-2,-3]]
        out = torch.squeeze(torch.transpose(torch.transpose(out,-1,-2),-2,-3))  # torch.squeeze(torch.transpos
#        out = torch.squeeze(torch.transpose(torch.transpose(model.decode(z).view(-1, 1, image_high, image_width, 3),-1,-2),-2,-3))  # torch.squeeze(torch.transpose(torch.transpose(x,-1,-2),-2,-3)) 
#       print('out1.shape',out.shape)
        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)))

        # Reconst Img
#        out, _, _ = model(x)
        out, _, _ = model(x_img)
#        print('out2.shape',out.shape)
#        x_concat = torch.cat([x.view(-1, 1, image_high, image_width), out.view(-1, 1, image_high, image_width)], dim=3)
        #x_concat = torch.cat([torch.squeeze(torch.transpose(torch.transpose(x.view(-1, 1, image_high, image_width, 3),-1,-2),-2,-3)), torch.squeeze(torch.transpose(torch.transpose(out.view(-1, 1, image_high, image_width, 3),-1,-2),-2,-3))], dim=3)
        x = x.view(-1, 1, image_high, image_width, 3)[:,:,:,:,[-1,-2,-3]]
        #x = x[:,:,:,:,[-1,-2,-3]]
        out = out.view(-1, 1, image_high, image_width, 3)[:,:,:,:,[-1,-2,-3]]
        #out = out[:,:,:,:,[-1,-2,-3]
        x_concat = torch.cat([torch.squeeze(torch.transpose(torch.transpose(x,-1,-2),-2,-3)), torch.squeeze(torch.transpose(torch.transpose(out,-1,-2),-2,-3))], dim=3)
        #print('x_concat.shape',out.shape)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))
