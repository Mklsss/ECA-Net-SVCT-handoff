# utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import torch
import config 
import os

def visualize_results(sparse_sino, full_sino, cnn_sino, fbp_from_sparse_sino, fbp_from_cnn, gen_output, ground_truth, epoch, save_path):

    sparse_sino_np = sparse_sino.cpu().detach().numpy()
    full_sino_np = full_sino.cpu().detach().numpy()
    cnn_sino_np = cnn_sino.cpu().detach().numpy()
    fbp_from_li_sino_np = fbp_from_sparse_sino.cpu().detach().numpy()
    fbp_from_cnn_np = fbp_from_cnn.cpu().detach().numpy()
    gen_output_np = gen_output.cpu().detach().numpy()
    ground_truth_np = ground_truth.cpu().detach().numpy()


    if sparse_sino_np.shape[1] == 1: sparse_sino_np = sparse_sino_np.squeeze(1)
    if full_sino_np.shape[1] == 1: full_sino_np = full_sino_np.squeeze(1)
    if cnn_sino_np.shape[1] == 1: cnn_sino_np = cnn_sino_np.squeeze(1)
    if fbp_from_li_sino_np.shape[1] == 1: fbp_from_li_sino_np = fbp_from_li_sino_np.squeeze(1)
    if fbp_from_cnn_np.shape[1] == 1: fbp_from_cnn_np = fbp_from_cnn_np.squeeze(1)
    if gen_output_np.shape[1] == 1: gen_output_np = gen_output_np.squeeze(1)
    if ground_truth_np.shape[1] == 1: ground_truth_np = ground_truth_np.squeeze(1)

    idx = 0 
    fig, axes = plt.subplots(2, 4, figsize=(20, 10)) 
    fig.suptitle(f'Epoch {epoch} Results', fontsize=16)
    vmin_sino, vmax_sino = (config.normalize_sino[0], config.normalize_sino[1]) if config.normalize_sino else (None, None)
    vmin_img, vmax_img = (config.normalize_img[0], config.normalize_img[1]) if config.normalize_img else (None, None)


    # --- 第1行：Sinograms & 来自线性插值的FBP ---
    im = axes[0, 0].imshow(sparse_sino_np[idx], cmap='gray', aspect='auto', vmin=vmin_sino, vmax=vmax_sino)
    axes[0, 0].set_title('Input Sparse Sinogram')
    axes[0, 0].set_xlabel('Detector Index'); axes[0, 0].set_ylabel('Angle Index')
    fig.colorbar(im, ax=axes[0,0], shrink=0.8)

    im = axes[0, 1].imshow(cnn_sino_np[idx], cmap='gray', aspect='auto', vmin=vmin_sino, vmax=vmax_sino)
    axes[0, 1].set_title('CNN Interpolated Sinogram')
    axes[0, 1].set_xlabel('Detector Index'); axes[0, 1].set_ylabel('Angle Index')
    fig.colorbar(im, ax=axes[0,1], shrink=0.8)

    im = axes[0, 2].imshow(full_sino_np[idx], cmap='gray', aspect='auto', vmin=vmin_sino, vmax=vmax_sino)
    axes[0, 2].set_title('GT Full Sinogram')
    axes[0, 2].set_xlabel('Detector Index'); axes[0, 2].set_ylabel('Angle Index')
    fig.colorbar(im, ax=axes[0,2], shrink=0.8)
    

    im = axes[0, 3].imshow(fbp_from_li_sino_np[idx], cmap='gray', vmin=vmin_img, vmax=vmax_img)
    axes[0, 3].set_title('FBP from Lin. Interp. Sino') # <<< 更新的标题
    axes[0, 3].axis('off')

    # --- 第2行：图像 ---
    im = axes[1, 0].imshow(fbp_from_cnn_np[idx], cmap='gray', vmin=vmin_img, vmax=vmax_img)
    axes[1, 0].set_title('FBP from CNN Sinogram')
    axes[1, 0].axis('off')

    im = axes[1, 1].imshow(gen_output_np[idx], cmap='gray', vmin=vmin_img, vmax=vmax_img)
    axes[1, 1].set_title('GAN Output Image')
    axes[1, 1].axis('off')

    im = axes[1, 2].imshow(ground_truth_np[idx], cmap='gray', vmin=vmin_img, vmax=vmax_img)
    axes[1, 2].set_title('Ground Truth CT Image')
    axes[1, 2].axis('off')

    diff_map = np.abs(ground_truth_np[idx] - gen_output_np[idx])
    im = axes[1, 3].imshow(diff_map, cmap='viridis', vmin=0, vmax=diff_map.max() if diff_map.max() > 1e-6 else 1.0)
    axes[1, 3].set_title('Difference (GT - GAN)')
    axes[1, 3].axis('off')
    fig.colorbar(im, ax=axes[1,3], shrink=0.8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    try:
        plt.savefig(save_path)
    except Exception as e:
        print(f"保存可视化结果 '{save_path}' 时出错: {e}")
    finally:
        plt.close(fig)

if __name__ == '__main__':
    print("测试使用线性插值Sinogram的FBP进行可视化")
    b, a_s, a_f, d_det, h_img, w_img = 1, len(config.SPARSE_VIEW_ANGLES), config.NUM_ANGLES_FULL, config.NUM_DETECTORS, config.IMAGE_SIZE, config.IMAGE_SIZE

    # 虚拟稀疏sinogram (仅在稀疏角度有值)
    s_sino_t_sparse_vals = torch.rand(b, 1, a_s, d_det) * 0.5 
    s_sino_t = torch.zeros(b, 1, a_f, d_det) # 完整形状，但内容稀疏
    s_sino_t[:, :, config.SPARSE_VIEW_ANGLES, :] = s_sino_t_sparse_vals

    f_sino_t = torch.rand(b, 1, a_f, d_det) 
    c_sino_t = torch.rand(b, 1, a_f, d_det) 
    fbp_li_sino_t = torch.rand(b, 1, h_img, w_img) # 来自线性插值稀疏sinogram的FBP
    fbp_cnn_t = torch.rand(b, 1, h_img, w_img)  
    gen_img_t = torch.rand(b, 1, h_img, w_img)  
    gt_img_t = torch.rand(b, 1, h_img, w_img)   

    test_save_path = os.path.join(config.OUTPUT_DIR, 'visualization_test_with_fbp_li_sino.png')
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    visualize_results(
        sparse_sino=s_sino_t,           
        full_sino=f_sino_t,
        cnn_sino=c_sino_t,
        fbp_from_sparse_sino=fbp_li_sino_t, 
        fbp_from_cnn=fbp_cnn_t,
        gen_output=gen_img_t,
        ground_truth=gt_img_t,
        epoch=1,
        save_path=test_save_path
    )
    print(f"已保存测试可视化结果到 {test_save_path}")
