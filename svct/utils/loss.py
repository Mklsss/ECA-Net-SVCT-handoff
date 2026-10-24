import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision.transforms import Normalize
import config

class ContentLoss(nn.Module):
    """
    Computes VGG perceptual loss (content loss).
    """
    def __init__(self, feature_layer_index=config.VGG_FEATURE_LAYER, device=config.DEVICE):
        super(ContentLoss, self).__init__()
        self.device = device
        vgg = vgg19(pretrained=True).features[:feature_layer_index + 1].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.loss = nn.MSELoss() 

        # VGG normalization 
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _preprocess(self, img_batch):
   
        if img_batch.shape[1] == 1:
            img_batch = img_batch.repeat(1, 3, 1, 1)

        # Normalize for VGG
        if img_batch.min() < -0.1: 
             img_batch = (img_batch + 1.0) / 2.0

        img_batch = self.normalize(img_batch)
        return img_batch


    def forward(self, generated_img, target_img):
       
        gen_features = self.vgg(self._preprocess(generated_img))
        target_features = self.vgg(self._preprocess(target_img))
    

        content_loss = self.loss(gen_features, target_features)
        return content_loss

def total_variation_loss(img_batch):

    # Pixel differences horizontally (wraps around)
    dh = torch.abs(img_batch[:, :, :, 1:] - img_batch[:, :, :, :-1])
    # Pixel differences vertically (wraps around)
    dv = torch.abs(img_batch[:, :, 1:, :] - img_batch[:, :, :-1, :])

    # Sum of absolute differences
    tv_loss = torch.sum(dh) + torch.sum(dv)

    # Normalize by batch size, channels, height, width
    b, c, h, w = img_batch.size()
    tv_loss = tv_loss / (b * c * h * w) # Normalize per pixel

    return tv_loss


if __name__ == '__main__':
    print("Testing Loss Functions")
    device = config.DEVICE
    dummy_img1 = torch.rand(4, 1, 256, 256).to(device) * 2 - 1 # Range [-1, 1]
    dummy_img2 = torch.rand(4, 1, 256, 256).to(device) * 2 - 1 # Range [-1, 1]

    # MSE Loss
    mse_loss_fn = nn.MSELoss()
    mse = mse_loss_fn(dummy_img1, dummy_img2)
    print(f"MSE Loss: {mse.item()}")

    # Content Loss
    content_loss_fn = ContentLoss(device=device)
    closs = content_loss_fn(dummy_img1, dummy_img2)
    print(f"Content Loss (VGG): {closs.item()}")

    # TV Loss
    tvloss = total_variation_loss(dummy_img1)
    print(f"Total Variation Loss: {tvloss.item()}")

    # Adversarial Loss 
    bce_logits_loss = nn.BCEWithLogitsLoss()
    dummy_logits = torch.randn(4, 1).to(device) # Output from Discriminator
    real_labels = torch.ones(4, 1).to(device)
    fake_labels = torch.zeros(4, 1).to(device)

    loss_real = bce_logits_loss(dummy_logits, real_labels)
    loss_fake = bce_logits_loss(dummy_logits, fake_labels)
    print(f"Example BCE Logits Loss (Real): {loss_real.item()}")
    print(f"Example BCE Logits Loss (Fake): {loss_fake.item()}")