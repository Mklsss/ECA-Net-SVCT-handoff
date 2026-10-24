import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def calculate_psnr(target, output, data_range=1.0):
  
    target_np = target.cpu().detach().numpy()
    output_np = output.cpu().detach().numpy()

    batch_psnr = []
    for i in range(target_np.shape[0]):
        t_img = target_np[i].squeeze() 
        o_img = output_np[i].squeeze()

        if np.any(np.isnan(o_img)) or np.any(np.isinf(o_img)):
            psnr = 0.0 # Assign a low value if output is invalid
        else:
             # Clip output to target range to avoid issues with PSNR calculation
             o_img = np.clip(o_img, target_np.min(), target_np.max())
             # Use data_range based on the *target* image's theoretical range
             psnr = compare_psnr(t_img, o_img, data_range=data_range)
        batch_psnr.append(psnr)

    return np.mean(batch_psnr)


def calculate_ssim(target, output, data_range=1.0, multichannel=False):
  
    target_np = target.cpu().detach().numpy()
    output_np = output.cpu().detach().numpy()

    if multichannel and target_np.shape[1] > 1:
         target_np = np.transpose(target_np, (0, 2, 3, 1))
         output_np = np.transpose(output_np, (0, 2, 3, 1))
    else: 
         target_np = target_np.squeeze(1)
         output_np = output_np.squeeze(1)


    batch_ssim = []
    for i in range(target_np.shape[0]):
        t_img = target_np[i]
        o_img = output_np[i]

        if np.any(np.isnan(o_img)) or np.any(np.isinf(o_img)):
             ssim = 0.0 # Assign a low value if output is invalid
        else:
             # Clip output
             o_img = np.clip(o_img, target_np.min(), target_np.max())
             ssim = compare_ssim(t_img, o_img, data_range=data_range, multichannel=multichannel,
                                win_size=7) # Use a smaller window size for 256x256 if needed
        batch_ssim.append(ssim)

    return np.mean(batch_ssim)

if __name__ == '__main__':
    print("Testing Metrics")
    # Create dummy images (e.g., normalized to [0, 1])
    img_a = torch.rand(4, 1, 256, 256)
    img_b = img_a * 0.8 + 0.1 # Slightly different image

    # Assume data range is 1.0 for [0, 1] normalization
    psnr = calculate_psnr(img_a, img_b, data_range=1.0)
    ssim = calculate_ssim(img_a, img_b, data_range=1.0, multichannel=False) # C=1 -> multichannel=False

    print(f"PSNR: {psnr:.4f}")
    print(f"SSIM: {ssim:.4f}")

    # Example with [-1, 1] range
    img_c = torch.rand(4, 1, 256, 256) * 2 - 1
    img_d = img_c * 0.8 - 0.1
    psnr_neg = calculate_psnr(img_c, img_d, data_range=2.0) # Data range is max-min = 1 - (-1) = 2
    ssim_neg = calculate_ssim(img_c, img_d, data_range=2.0, multichannel=False)
    print(f"PSNR ([-1, 1]): {psnr_neg:.4f}")
    print(f"SSIM ([-1, 1]): {ssim_neg:.4f}")
