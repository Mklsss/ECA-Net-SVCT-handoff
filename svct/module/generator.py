import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config 

class ECABlock(nn.Module):

    def __init__(self, channel, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel 

    def forward(self, x):
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class ResidualBlockWithECA(nn.Module):
    def __init__(self, in_features, out_features, bottleneck_expansion=4, use_eca=False, eca_k_size=3, stride=1):
        super(ResidualBlockWithECA, self).__init__()
        self.use_eca = use_eca
        bottleneck_features = in_features // bottleneck_expansion

        self.conv1 = nn.Conv2d(in_features, bottleneck_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_features)
        
        self.conv2 = nn.Conv2d(bottleneck_features, bottleneck_features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_features)

        self.conv3 = nn.Conv2d(bottleneck_features, out_features, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_features)
        
        self.relu = nn.ReLU(inplace=True)

        if self.use_eca:
            self.eca = ECABlock(out_features, k_size=eca_k_size)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features)
            )

    def forward(self, x):
        identity_res_block = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out) 

        if self.use_eca:
            out = self.eca(out)

        out += self.shortcut(identity_res_block) 
        out = self.relu(out) 
        return out

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,
                 features=config.GEN_FEATURES,
                 num_blocks=config.GEN_NUM_BLOCKS,
                 bottleneck_expansion=config.GEN_BOTTLENECK_EXPANSION,
                 use_eca=config.USE_ECA_IN_GENERATOR,
                 eca_k_size=config.ECA_KERNEL_SIZE):
        super(Generator, self).__init__()

        assert in_channels == out_channels, \
            "For this simplified Generator with global skip, in_channels must equal out_channels."

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

        blocks = []
        current_features = features
        for _ in range(num_blocks):
            blocks.append(ResidualBlockWithECA(current_features, features, 
                                               bottleneck_expansion=bottleneck_expansion,
                                               use_eca=use_eca, eca_k_size=eca_k_size))
        self.residual_blocks = nn.Sequential(*blocks)

        self.post_res_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        identity_global_skip = x

        out = self.initial_conv(x)
        out_res_blocks = self.residual_blocks(out)
        out_post_conv = self.post_res_conv(out_res_blocks)
        
        features_before_activation = self.final_conv(out_post_conv)
        
        summed_features = features_before_activation + identity_global_skip

        if config.normalize_img == (-1, 1):
             final_image = torch.tanh(summed_features) 
        elif config.normalize_img == (0, 1):
             final_image = torch.sigmoid(summed_features) 
        else:
            final_image = summed_features 

        return final_image

if __name__ == '__main__':
    print("--- Testing Generator (Simplified Global Skip) ---")
    dummy_fbp_image = torch.randn(2, 1, config.IMAGE_SIZE, config.IMAGE_SIZE) # Assuming in_channels=1, out_channels=1
    model = Generator(in_channels=1, out_channels=1,
                      features=config.GEN_FEATURES,
                      num_blocks=config.GEN_NUM_BLOCKS,
                      bottleneck_expansion=config.GEN_BOTTLENECK_EXPANSION,
                      use_eca=config.USE_ECA_IN_GENERATOR,
                      eca_k_size=config.ECA_KERNEL_SIZE)
    # print(model)
    output_image = model(dummy_fbp_image)
    print("Input shape:", dummy_fbp_image.shape)
    print("Output shape:", output_image.shape)
    assert dummy_fbp_image.shape == output_image.shape
    print("Generator test (simplified global skip) passed.")
