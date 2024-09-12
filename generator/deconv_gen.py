import torch.nn as nn
import pytorch_lightning as pl
import torch
import torch.nn.functional as F


# class Generator(nn.Module):
#     def __init__(self, z_shape, output_channels, grid_size=8):
#         super(Generator, self).__init__()
#         self.z_size = z_shape
#         self.output_channels = output_channels
#         self.grid_size = grid_size

#         # Calculate the size of intermediate layers based on grid_size
#         intermediate_size = grid_size // 4  # This will be 2 if grid_size is 8

#         self.main = nn.Sequential(
#             nn.ConvTranspose2d(self.z_size, 256, kernel_size=(intermediate_size, intermediate_size), stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),

#             nn.ConvTranspose2d(128, self.output_channels, kernel_size=4, stride=2, padding=1, bias=False)  # Output: (batch_size, output_channels, 8, 8)
#         )

#         self.output = nn.ReLU()  # Use ReLU to ensure positive outputs

#     def forward(self, z):
#         # Reshape the input noise vector z to match the starting dimensions
#         x = z.reshape(-1, self.z_size, 1, 1)
#         x = self.main(x)
#         x = self.output(x)
#         return x


# class Discriminator(nn.Module):
#     def __init__(self, input_channels=2, grid_size=8, dropout=0.3):
#         super(Discriminator, self).__init__()
#         self.grid_size = grid_size
#         ndf = 64  # Number of filters in the first layer
        
#         self.main = nn.Sequential(
#             nn.Conv2d(input_channels, ndf, 3, 1, 1, bias=False),  # Output: (batch_size, 64, grid_size/2, grid_size/2)
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(dropout),

#             nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False),  # Output: (batch_size, 128, grid_size/2, grid_size/2)
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(dropout),

#             nn.Conv2d(ndf * 2, 1, 3, 2, 1, bias=False),  # Output: (batch_size, 1, 4, 4)
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(1, 1, 4, 1, 0, bias=False),  # Output: (batch_size, 1, 1, 1)
#             nn.Sigmoid()  # Output a single probability (real vs. fake)
#         )

#     def forward(self, input):
#         input = input.float()
#         return self.main(input).view(-1, 1).squeeze(1)

class Generator(nn.Module):
    def __init__(self, z_shape, output_channel=2, num_classes=6, grid_size=8):
        super(Generator, self).__init__()
        self.z_size = z_shape
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.output_channel = output_channel


        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.z_size, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # nn.ConvTranspose2d(128, self.num_classes, kernel_size=4, stride=2, padding=1, bias=False)
            nn.ConvTranspose2d(128, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)

        )

    def forward(self, z):
        #(batch_size, z_size, 1, 1)
        x = z.reshape(-1, self.z_size, 1, 1)
        x = self.main(x)        
        x = nn.Softmax(dim=1)(x)  # (batch_size, num_classes, 8, 8)
        # x = torch.argmax(x, dim=2)
        return x # (32,2,6,8,8)

class Discriminator(nn.Module):
    def __init__(self, input_channels=6, grid_size=8, dropout=0.3):
        super(Discriminator, self).__init__()
        ndf = 64  # Number of filters in the first layer

        self.main = nn.Sequential(
            nn.Conv2d(input_channels, ndf, 3, 1, 1, bias=False),  # Adjusted input_channels to 6
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Conv2d(128, 1, kernel_size=8, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.shape[1:] == (8, 8):
            # Reshape input from (32, 2, 8, 8) to (32 * 2 * 8 * 8)
            reshaped_input = input.view(-1)  # Flatten 

            # Apply one-hot encoding
            one_hot_encoded = F.one_hot(reshaped_input, num_classes=6)  

            # Reshape back to (32, 2, 6, 8, 8)
            input = one_hot_encoded.view(input.shape[0], 8, 8, 6).permute(0, 3, 1, 2)  # Shape (32, 6, 8, 8)

        input = input.float()  
        output = self.main(input)  
        return output.view(-1)  