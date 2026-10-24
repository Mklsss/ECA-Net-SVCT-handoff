import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels=16, out_channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=3, padding=1)
      
        if in_channels != out_channels:
             self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
             self.skip_conv = None


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        if self.skip_conv is not None:
            identity = self.skip_conv(identity)

        out += identity # Add skip connection

        return out

class ProjectionCNN(nn.Module):
    def __init__(self, in_channels=1, num_blocks=4, features=64, bottleneck_features=16):
        super(ProjectionCNN, self).__init__()
        self.in_channels = in_channels
        self.features = features

        # Initial convolution layer
        self.conv_in = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.relu_in = nn.ReLU(inplace=True)

        # Residual blocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(features, bottleneck_features, features))
        self.blocks = nn.Sequential(*blocks)

        # Final convolution layer
        self.conv_out = nn.Conv2d(features, features, kernel_size=3, padding=1) # Intermediate conv
        self.relu_out1 = nn.ReLU(inplace=True) # Added based on diagram structure
        self.conv_final = nn.Conv2d(features, in_channels, kernel_size=3, padding=1) # To output channels


    def forward(self, x):
        identity = x # Store original input for global skip connection

        out = self.relu_in(self.conv_in(x))
        out = self.blocks(out)
        out = self.relu_out1(self.conv_out(out))
        out = self.conv_final(out)

        out = out + identity

        return out

if __name__ == '__main__':

    dummy_sparse_sinogram = torch.randn(4, 1, 360, 357) 
    model = ProjectionCNN(in_channels=1)
    output_sinogram = model(dummy_sparse_sinogram)
    print("Input shape:", dummy_sparse_sinogram.shape)
    print("Output shape:", output_sinogram.shape)  