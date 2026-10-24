import torch
import torch.nn as nn
import config 

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, img_size=config.IMAGE_SIZE, features_start=config.DISC_FEATURES_START):
        super(Discriminator, self).__init__()
        self.img_size = img_size

        def discriminator_block(in_filters, out_filters, stride, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=stride, padding=1, bias=not normalize)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        f = features_start # 64
        self.conv_blocks = nn.Sequential(
            # Block 1: in_channels -> f, K=3, S=1, BN, LeakyReLU
            *discriminator_block(in_channels, f, stride=1),      
            # Block 2: f -> f, K=3, S=2, BN, LeakyReLU
            *discriminator_block(f, f, stride=2),               
            # Block 3: f -> f*2, K=3, S=1, BN, LeakyReLU
            *discriminator_block(f, f*2, stride=1),             
            # Block 4: f*2 -> f*2, K=3, S=2, BN, LeakyReLU
            *discriminator_block(f*2, f*2, stride=2),           
            # Block 5: f*2 -> f*4, K=3, S=1, BN, LeakyReLU
            *discriminator_block(f*2, f*4, stride=1),           
            # Block 6: f*4 -> f*4, K=3, S=2, BN, LeakyReLU
            *discriminator_block(f*4, f*4, stride=2)            
        )

        # 动态计算展平后的特征大小
        self.final_conv_output_size = img_size // 8 
        linear_input_features = (f*4) * self.final_conv_output_size * self.final_conv_output_size
        #全连接层序列
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_features, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, img):
        x = self.conv_blocks(img)
        x = self.fc_layers(x)
        return x

if __name__ == '__main__':
    print("--- Testing Discriminator ---")
    dummy_image = torch.randn(2, 1, config.IMAGE_SIZE, config.IMAGE_SIZE) # B, C, H, W
    model = Discriminator(in_channels=1, img_size=config.IMAGE_SIZE, features_start=config.DISC_FEATURES_START)
    print(model)
    probability = model(dummy_image)
    print("Input shape:", dummy_image.shape)
    print("Output shape (Logits):", probability.shape)


    print("Discriminator test passed.")

