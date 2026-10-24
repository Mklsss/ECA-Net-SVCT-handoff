# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import argparse
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d  # <<< 添加此导入
import swanlab
# 初始化一个新的swanlab run类来跟踪这个脚本
swanlab.init(
  # 设置将记录此次运行的项目信息
  project="svct",
  workspace="mkls",
  # 跟踪超参数和运行元数据
  config={
    "BATCH_SIZE": 8,
    "LEARNING_RATE_CNN": 1e-4,
    "LEARNING_RATE_G": 5e-5,
    "LEARNING_RATE_D": 5e-6,
    "EPOCHS_CNN": 50,
    "EPOCHS_GAN": 200,
    "PRETRAIN_GAN_MSE_EPOCHS": 30,
    "ECA": True
      
  }
)




# 导入项目模块
import config
from dataset import CTDataset
from module import ProjectionCNN, Generator, Discriminator
from utils import (
    fbp_reconstruct,
    ContentLoss, total_variation_loss,
    calculate_psnr, calculate_ssim,
    visualize_results
)


# <<< 开始：新的线性插值辅助函数 >>>
def linear_interpolate_sinogram_torch(sparse_sino_tensor, sparse_angles_indices, full_num_angles, device):
    """
    沿角度维度对稀疏正弦图张量进行线性插值。

    参数:
        sparse_sino_tensor (torch.Tensor): 稀疏正弦图 (B, 1, NumFullAngles, NumDetectors)。
                                          假设非稀疏角度处为0或将被忽略。
        sparse_angles_indices (np.ndarray): 包含稀疏视图所在位置索引的一维数组。
        full_num_angles (int): 密集正弦图中的总角度数。
        device (torch.device): 输出张量所在的设备。

    返回:
        torch.Tensor: 线性插值后的正弦图 (B, 1, NumFullAngles, NumDetectors)。
    """
    batch_size, _, _, num_detectors = sparse_sino_tensor.shape
    sparse_sino_np = sparse_sino_tensor.cpu().numpy()  # (B, 1, A_full, D)

    interpolated_sinos_list = []

    # 定义插值的x坐标（所有角度）
    xp = np.arange(full_num_angles)

    for b_idx in range(batch_size):
        # current_sparse_sino是 (1, A_full, D)，我们从中提取已知值
        # known_values是 (NumSparseAngles, NumDetectors)
        known_projection_values = sparse_sino_np[b_idx, 0, sparse_angles_indices, :]

        interpolated_sino_single_batch = np.zeros((full_num_angles, num_detectors), dtype=np.float32)

        for det_idx in range(num_detectors):
            # fp_detector是 (NumSparseAngles,)
            fp_detector = known_projection_values[:, det_idx]

            # 对此探测器列进行一维插值
            # sparse_angles_indices 是已知数据点的x坐标
            # fp_detector 是已知数据点的y坐标
            # xp 是我们想要插值的x坐标
            interp_func = interp1d(sparse_angles_indices, fp_detector, kind='linear',
                                   bounds_error=False, fill_value="extrapolate")
            # fill_value也可以是一个元组 (val_at_min_angle, val_at_max_angle)
            # 或者如果可接受，简单使用 "extrapolate"

            interpolated_sino_single_batch[:, det_idx] = interp_func(xp)

        interpolated_sinos_list.append(interpolated_sino_single_batch)

    interpolated_sinos_np = np.stack(interpolated_sinos_list, axis=0)  # (B, A_full, D)
    interpolated_sinos_tensor = torch.from_numpy(interpolated_sinos_np).unsqueeze(1).to(device)  # (B, 1, A_full, D)

    return interpolated_sinos_tensor


# <<< 结束：新的辅助函数 >>>


def train_cnn(cnn_model, train_loader, optimizer, criterion, device, epochs):
    """训练投影CNN。"""
    print("\n--- 开始投影CNN训练 ---")
    cnn_model.train()
    history = {'epoch': [], 'loss': []}

    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        for batch in progress_bar:
            sparse_sinogram = batch['sparse_sinogram'].to(device)
            full_sinogram = batch['full_sinogram'].to(device)

            optimizer.zero_grad()
            interpolated_sinogram = cnn_model(sparse_sinogram)
            loss = criterion(interpolated_sinogram, full_sinogram)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.6f} - Time: {epoch_time:.2f}s")

        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_epoch_loss)

        if (epoch + 1) % config.SAVE_INTERVAL == 0 or (epoch + 1) == epochs:
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            save_path = os.path.join(config.OUTPUT_DIR, f'projection_cnn_epoch_{epoch + 1}.pth')
            torch.save(cnn_model.state_dict(), save_path)
            print(f"已保存CNN模型检查点到 {save_path}")

    print("--- 投影CNN训练完成 ---")
    return history


def train_gan(cnn_model, generator, discriminator, train_loader, optim_g, optim_d, device, epochs, pretrain_mse_epochs):
    """训练GAN（生成器和判别器）。"""
    print("\n--- 开始GAN训练 ---")
    cnn_model.eval()
    generator.train()
    discriminator.train()

    criterion_gan = nn.BCEWithLogitsLoss().to(device)
    criterion_pixel = nn.MSELoss().to(device)
    criterion_content = ContentLoss(device=device)

    real_label_base = torch.ones((config.BATCH_SIZE, 1), device=device, dtype=torch.float32)
    fake_label_base = torch.zeros((config.BATCH_SIZE, 1), device=device, dtype=torch.float32)
    if config.LABEL_SMOOTHING_REAL < 1.0:
        real_label_base = real_label_base * config.LABEL_SMOOTHING_REAL
    if config.LABEL_SMOOTHING_FAKE > 0.0:
        fake_label_base = fake_label_base + config.LABEL_SMOOTHING_FAKE

    history = {'epoch': [], 'g_loss': [], 'd_loss': [], 'psnr': [], 'ssim': []}

    if pretrain_mse_epochs > 0:
        print(f"\n--- 使用MSE预训练生成器 {pretrain_mse_epochs} 个epoch ---")
        optim_g_pretrain = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE_G,
                                      betas=(config.ADAM_BETA1, config.ADAM_BETA2))
        for epoch in range(pretrain_mse_epochs):
            epoch_g_loss_pre = 0.0
            progress_bar = tqdm(train_loader, desc=f"预训练 Epoch {epoch + 1}/{pretrain_mse_epochs}", leave=True)
            for batch in progress_bar:
                with torch.no_grad():
                    sparse_sinogram = batch['sparse_sinogram'].to(device)
                    interpolated_sino_cnn = cnn_model(sparse_sinogram)  # 使用CNN插值的sinogram

                    # 对CNN插值的sinogram使用全角度集进行FBP重建
                    fbp_image = fbp_reconstruct(interpolated_sino_cnn, config.ANGLES_RAD, output_size=config.IMAGE_SIZE)

                    if fbp_image.numel() > 0:
                        fbp_min = fbp_image.min()
                        fbp_max = fbp_image.max()
                        if fbp_max > fbp_min:
                            fbp_image = (fbp_image - fbp_min) / (fbp_max - fbp_min)
                            if config.normalize_img == (-1, 1):
                                fbp_image = fbp_image * 2.0 - 1.0
                        else:
                            fbp_image = torch.full_like(fbp_image,
                                                        0.0 if config.normalize_img[0] == -1 else config.normalize_img[
                                                            0])

                ct_label = batch['ct_label'].to(device)
                optim_g_pretrain.zero_grad()
                generated_image = generator(fbp_image)  # 生成器输入是来自CNN的归一化FBP
                loss_g_mse = criterion_pixel(generated_image, ct_label)
                loss_g_mse.backward()
                optim_g_pretrain.step()
                epoch_g_loss_pre += loss_g_mse.item()
                progress_bar.set_postfix(mse_loss=f"{loss_g_mse.item():.4f}")
            avg_epoch_g_loss_pre = epoch_g_loss_pre / len(train_loader)
            print(f"预训练 Epoch {epoch + 1}/{pretrain_mse_epochs} - 生成器 MSE Loss: {avg_epoch_g_loss_pre:.6f}")
            # 向swanlab上传训练指标
            swanlab.log({"生成器 MSE Loss": avg_epoch_g_loss_pre})
        print("--- 生成器预训练完成 ---")

    print("\n--- 开始对抗性训练 ---")
    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f"GAN Epoch {epoch + 1}/{epochs}", leave=True)

        for i, batch in enumerate(progress_bar):
            sparse_sinogram_orig = batch['sparse_sinogram'].to(device)
            ct_label = batch['ct_label'].to(device)
            full_sinogram_gt = batch['full_sinogram'].to(device)  # 用于可视化

            with torch.no_grad():
                interpolated_sino_cnn = cnn_model(sparse_sinogram_orig)
                fbp_from_cnn_output_raw = fbp_reconstruct(interpolated_sino_cnn, config.ANGLES_RAD,
                                                          output_size=config.IMAGE_SIZE)

                fbp_from_cnn_output_norm = torch.zeros_like(fbp_from_cnn_output_raw)
                if fbp_from_cnn_output_raw.numel() > 0:
                    fbp_min_cnn = fbp_from_cnn_output_raw.min()
                    fbp_max_cnn = fbp_from_cnn_output_raw.max()
                    if fbp_max_cnn > fbp_min_cnn:
                        fbp_from_cnn_output_norm = (fbp_from_cnn_output_raw - fbp_min_cnn) / (fbp_max_cnn - fbp_min_cnn)
                        if config.normalize_img == (-1, 1):
                            fbp_from_cnn_output_norm = fbp_from_cnn_output_norm * 2.0 - 1.0
                    else:
                        fbp_from_cnn_output_norm = torch.full_like(fbp_from_cnn_output_raw,
                                                                   float(config.normalize_img[0]))

                bp_from_li_sino_norm = torch.empty(0)  # 或者 None，然后在 visualize_results 中处理
                if device.type != 'mps':  # .empty() is not supported on MPS
                    fbp_from_li_sino_norm = torch.empty(0, device=device)
                else:  # Fallback for MPS or if you prefer a zero tensor
                    fbp_from_li_sino_norm = torch.zeros_like(fbp_from_cnn_output_norm)

            real_image = ct_label
            current_batch_size = real_image.size(0)
            current_real_label = real_label_base[
                                 :current_batch_size] if current_batch_size != config.BATCH_SIZE else real_label_base
            current_fake_label = fake_label_base[
                                 :current_batch_size] if current_batch_size != config.BATCH_SIZE else fake_label_base

            # --- 训练判别器 ---
            optim_d.zero_grad()
            fake_image = generator(fbp_from_cnn_output_norm)  # G的输入是来自CNN sinogram的FBP

            pred_real = discriminator(real_image)
            loss_d_real = criterion_gan(pred_real, current_real_label)
            pred_fake = discriminator(fake_image.detach())
            loss_d_fake = criterion_gan(pred_fake, current_fake_label)
            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            if config.GRAD_CLIP_VALUE > 0:
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), config.GRAD_CLIP_VALUE)
            optim_d.step()

            # --- 训练生成器 ---
            loss_g_val = 0.0  # 初始化
            if (i + 1) % config.D_UPDATE_FREQ == 0:
                optim_g.zero_grad()
                # fake_image 已生成
                pred_fake_g = discriminator(fake_image)
                loss_g_adv = criterion_gan(pred_fake_g, current_real_label)
                loss_g_pixel = criterion_pixel(fake_image, real_image)
                loss_g_content = criterion_content(fake_image, real_image)
                loss_g_tv = total_variation_loss(fake_image)

                loss_g = (config.LAMBDA_MSE * loss_g_pixel +
                          config.LAMBDA_ADV * loss_g_adv +
                          config.LAMBDA_CONTENT * loss_g_content +
                          config.LAMBDA_TV * loss_g_tv)
                loss_g.backward()
                if config.GRAD_CLIP_VALUE > 0:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), config.GRAD_CLIP_VALUE)
                optim_g.step()
                loss_g_val = loss_g.item()

            epoch_d_loss += loss_d.item()
            epoch_g_loss += loss_g_val

            data_range = 2.0 if config.normalize_img == (-1, 1) else 1.0
            psnr_val = calculate_psnr(real_image, fake_image.detach(), data_range=data_range)
            ssim_val = calculate_ssim(real_image, fake_image.detach(), data_range=data_range, multichannel=False)
            epoch_psnr += psnr_val
            epoch_ssim += ssim_val

            progress_bar.set_postfix(
                G_loss=f"{loss_g_val:.4f}", D_loss=f"{loss_d.item():.4f}",
                PSNR=f"{psnr_val:.2f}", SSIM=f"{ssim_val:.3f}"
            )

            if i == 0 and ((epoch + 1) % config.VISUALIZE_INTERVAL == 0 or (epoch + 1) == epochs):
                print(f"\n为epoch {epoch + 1} 生成可视化结果...")
                with torch.no_grad():
                    linearly_interpolated_sino = linear_interpolate_sinogram_torch(
                        sparse_sinogram_orig,
                        config.SPARSE_VIEW_ANGLES,
                        config.NUM_ANGLES_FULL,
                        device
                    )
                    fbp_from_li_sino_raw = fbp_reconstruct(linearly_interpolated_sino, config.ANGLES_RAD,
                                                           output_size=config.IMAGE_SIZE)
                    # 重新计算 fbp_from_li_sino_norm (之前可能是占位符)
                    fbp_from_li_sino_norm = torch.zeros_like(fbp_from_li_sino_raw)
                    if fbp_from_li_sino_raw.numel() > 0:
                        fbp_min_li = fbp_from_li_sino_raw.min()
                        fbp_max_li = fbp_from_li_sino_raw.max()
                        if fbp_max_li > fbp_min_li:
                            fbp_from_li_sino_norm = (fbp_from_li_sino_raw - fbp_min_li) / (fbp_max_li - fbp_min_li)
                            if config.normalize_img == (-1, 1):
                                fbp_from_li_sino_norm = fbp_from_li_sino_norm * 2.0 - 1.0
                        else:
                            fbp_from_li_sino_norm = torch.full_like(fbp_from_li_sino_raw,
                                                                    float(config.normalize_img[0]))
                # <<< 结束修改 >>>

                os.makedirs(os.path.join(config.OUTPUT_DIR, 'gan_visuals'), exist_ok=True)
                save_vis_path = os.path.join(config.OUTPUT_DIR, 'gan_visuals', f'epoch_{epoch + 1}_results.png')
                visualize_results(
                    sparse_sino=sparse_sinogram_orig,
                    full_sino=full_sinogram_gt,
                    cnn_sino=interpolated_sino_cnn,
                    fbp_from_sparse_sino=fbp_from_li_sino_norm,  # <<< 使用新计算的值
                    fbp_from_cnn=fbp_from_cnn_output_norm,
                    gen_output=fake_image.detach(),
                    ground_truth=real_image,
                    epoch=(epoch + 1),
                    save_path=save_vis_path
                )
                print(f"已保存可视化结果到 {save_vis_path}")

        avg_epoch_g_loss = epoch_g_loss / (
                    len(train_loader) / config.D_UPDATE_FREQ) if config.D_UPDATE_FREQ > 0 else epoch_g_loss / len(
            train_loader)
        avg_epoch_d_loss = epoch_d_loss / len(train_loader)
        avg_epoch_psnr = epoch_psnr / len(train_loader)
        avg_epoch_ssim = epoch_ssim / len(train_loader)
        epoch_time = time.time() - start_time

        print(
            f"GAN Epoch {epoch + 1}/{epochs} - Time: {epoch_time:.2f}s\n"
            f"  Avg G Loss: {avg_epoch_g_loss:.6f}, Avg D Loss: {avg_epoch_d_loss:.6f}\n"
            f"  Avg PSNR: {avg_epoch_psnr:.4f}, Avg SSIM: {avg_epoch_ssim:.4f}"
        )
        swanlab.log({"Avg G Loss": avg_epoch_g_loss, "Avg D Loss": avg_epoch_d_loss, "Avg PSNR":avg_epoch_psnr, "Avg SSIM":avg_epoch_ssim})
        
        history['epoch'].append(epoch + 1)
        history['g_loss'].append(avg_epoch_g_loss)
        history['d_loss'].append(avg_epoch_d_loss)
        history['psnr'].append(avg_epoch_psnr)
        history['ssim'].append(avg_epoch_ssim)

        # ===== 每个 epoch 完成后跑一次测试集评估 =====
        print(f"\n=== Running test after Epoch {epoch + 1} ===")
        test_psnr, test_ssim = test(cnn_model, generator, test_loader, device)
        print(f"[Test Results]  PSNR: {test_psnr:.4f}, SSIM: {test_ssim:.4f}\n")

        swanlab.log({"[Test Results]  PSNR:": test_psnr, "[Test Results]  SSIM": test_ssim})
        

        if (epoch + 1) % config.SAVE_INTERVAL == 0 or (epoch + 1) == epochs:
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            save_path_g = os.path.join(config.OUTPUT_DIR, f'generator_epoch_{epoch + 1}.pth')
            save_path_d = os.path.join(config.OUTPUT_DIR, f'discriminator_epoch_{epoch + 1}.pth')
            torch.save(generator.state_dict(), save_path_g)
            torch.save(discriminator.state_dict(), save_path_d)
            print(f"已保存GAN模型检查点到 {config.OUTPUT_DIR}")

    print("--- GAN训练完成 ---")
    # [可选] 完成训练，这在notebook环境中是必要的
    swanlab.finish()
    return history


def test(cnn_model, generator, test_loader, device):
    """在测试集上测试完整流程 (CNN + FBP + 生成器)。"""
    print("\n--- 开始测试阶段 ---")
    cnn_model.eval()
    generator.eval()

    total_psnr_fbp_cnn = 0.0
    total_ssim_fbp_cnn = 0.0
    total_psnr_gan = 0.0
    total_ssim_gan = 0.0
    total_psnr_fbp_li = 0.0  # <<< 用于线性插值sinogram的FBP
    total_ssim_fbp_li = 0.0  # <<< 用于线性插值sinogram的FBP

    start_time = time.time()
    progress_bar = tqdm(test_loader, desc="测试中", leave=False)

    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            sparse_sinogram_orig = batch['sparse_sinogram'].to(device)
            ct_label = batch['ct_label'].to(device)

            # 1. CNN 插值
            interpolated_sino_cnn = cnn_model(sparse_sinogram_orig)

            # 2. 从CNN输出进行FBP重建
            fbp_from_cnn_output_raw = fbp_reconstruct(interpolated_sino_cnn, config.ANGLES_RAD,
                                                      output_size=config.IMAGE_SIZE)
            fbp_from_cnn_output_norm = torch.zeros_like(fbp_from_cnn_output_raw)
            if fbp_from_cnn_output_raw.numel() > 0:
                fbp_min_cnn, fbp_max_cnn = fbp_from_cnn_output_raw.min(), fbp_from_cnn_output_raw.max()
                if fbp_max_cnn > fbp_min_cnn:
                    fbp_from_cnn_output_norm = (fbp_from_cnn_output_raw - fbp_min_cnn) / (fbp_max_cnn - fbp_min_cnn)
                    if config.normalize_img == (-1, 1): fbp_from_cnn_output_norm = fbp_from_cnn_output_norm * 2.0 - 1.0
                else:
                    fbp_from_cnn_output_norm = torch.full_like(fbp_from_cnn_output_raw, float(config.normalize_img[0]))

            # 3. 生成器增强
            gan_output_image = generator(fbp_from_cnn_output_norm)

            # --- 从线性插值的稀疏sinogram进行FBP，用于基线比较 ---
            linearly_interpolated_sino = linear_interpolate_sinogram_torch(
                sparse_sinogram_orig,
                config.SPARSE_VIEW_ANGLES,
                config.NUM_ANGLES_FULL,
                device
            )
            fbp_from_li_sino_raw = fbp_reconstruct(linearly_interpolated_sino, config.ANGLES_RAD,
                                                   output_size=config.IMAGE_SIZE)
            fbp_from_li_sino_norm = torch.zeros_like(fbp_from_li_sino_raw)
            if fbp_from_li_sino_raw.numel() > 0:
                fbp_min_li, fbp_max_li = fbp_from_li_sino_raw.min(), fbp_from_li_sino_raw.max()
                if fbp_max_li > fbp_min_li:
                    fbp_from_li_sino_norm = (fbp_from_li_sino_raw - fbp_min_li) / (fbp_max_li - fbp_min_li)
                    if config.normalize_img == (-1, 1): fbp_from_li_sino_norm = fbp_from_li_sino_norm * 2.0 - 1.0
                else:
                    fbp_from_li_sino_norm = torch.full_like(fbp_from_li_sino_raw, float(config.normalize_img[0]))

            # --- 计算指标 ---
            data_range = 2.0 if config.normalize_img == (-1, 1) else 1.0
            mch = False if ct_label.ndim == 4 and ct_label.size(1) == 1 else True

            # 线性插值sinogram的FBP vs 真实CT的指标
            psnr_fbp_li_val = calculate_psnr(ct_label, fbp_from_li_sino_norm, data_range=data_range)
            ssim_fbp_li_val = calculate_ssim(ct_label, fbp_from_li_sino_norm, data_range=data_range, multichannel=mch)
            total_psnr_fbp_li += psnr_fbp_li_val
            total_ssim_fbp_li += ssim_fbp_li_val

            # CNN sinogram的FBP vs 真实CT的指标
            psnr_fbp_cnn_val = calculate_psnr(ct_label, fbp_from_cnn_output_norm, data_range=data_range)
            ssim_fbp_cnn_val = calculate_ssim(ct_label, fbp_from_cnn_output_norm, data_range=data_range,
                                              multichannel=mch)
            total_psnr_fbp_cnn += psnr_fbp_cnn_val
            total_ssim_fbp_cnn += ssim_fbp_cnn_val

            # 最终GAN输出 vs 真实CT的指标
            psnr_gan_val = calculate_psnr(ct_label, gan_output_image, data_range=data_range)
            ssim_gan_val = calculate_ssim(ct_label, gan_output_image, data_range=data_range, multichannel=mch)
            total_psnr_gan += psnr_gan_val
            total_ssim_gan += ssim_gan_val

            progress_bar.set_postfix(
                PSNR_GAN=f"{psnr_gan_val:.2f}", SSIM_GAN=f"{ssim_gan_val:.3f}",
                PSNR_FBP_CNN=f"{psnr_fbp_cnn_val:.2f}",
                PSNR_FBP_LI=f"{psnr_fbp_li_val:.2f}"  # <<< 显示新指标
            )

            if i < config.NUM_TEST_VISUALS:
                os.makedirs(os.path.join(config.OUTPUT_DIR, 'test_results'), exist_ok=True)
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 增加figsize
                title_fontsize = 9
                vmin_val, vmax_val = (config.normalize_img[0], config.normalize_img[1]) if config.normalize_img else (
                None, None)

                axes[0].imshow(fbp_from_li_sino_norm[0, 0].cpu().numpy(), cmap='gray', vmin=vmin_val, vmax=vmax_val)
                axes[0].set_title(f'FBP Lin.Interp Sino\n(PSNR:{psnr_fbp_li_val:.2f})', fontsize=title_fontsize)
                axes[0].axis('off')

                axes[1].imshow(fbp_from_cnn_output_norm[0, 0].cpu().numpy(), cmap='gray', vmin=vmin_val, vmax=vmax_val)
                axes[1].set_title(f'FBP from CNN Sino\n(PSNR:{psnr_fbp_cnn_val:.2f})', fontsize=title_fontsize)
                axes[1].axis('off')

                axes[2].imshow(gan_output_image[0, 0].cpu().numpy(), cmap='gray', vmin=vmin_val, vmax=vmax_val)
                axes[2].set_title(f'GAN Output\n(PSNR:{psnr_gan_val:.2f})', fontsize=title_fontsize)
                axes[2].axis('off')

                axes[3].imshow(ct_label[0, 0].cpu().numpy(), cmap='gray', vmin=vmin_val, vmax=vmax_val)
                axes[3].set_title('Ground Truth', fontsize=title_fontsize)
                axes[3].axis('off')

                plt.tight_layout(pad=0.5)
                plt.savefig(os.path.join(config.OUTPUT_DIR, 'test_results', f'test_sample_{i}.png'))
                plt.close(fig)
                print(
                    f"已保存样本 {i} 的测试可视化结果到 {os.path.join(config.OUTPUT_DIR, 'test_results', f'test_sample_{i}.png')}")

    test_time = time.time() - start_time
    num_batches = len(test_loader)
    avg_psnr_fbp_li = total_psnr_fbp_li / num_batches
    avg_ssim_fbp_li = total_ssim_fbp_li / num_batches
    avg_psnr_fbp_cnn = total_psnr_fbp_cnn / num_batches
    avg_ssim_fbp_cnn = total_ssim_fbp_cnn / num_batches
    avg_psnr_gan = total_psnr_gan / num_batches
    avg_ssim_gan = total_ssim_gan / num_batches

    print("--- 测试完成 ---")
    print(f"测试时间: {test_time:.2f}s")
    print(f"平均 PSNR (FBP from Linearly Interpolated Sino): {avg_psnr_fbp_li:.4f}")  # <<< 新增
    print(f"平均 SSIM (FBP from Linearly Interpolated Sino): {avg_ssim_fbp_li:.4f}")  # <<< 新增
    print(f"平均 PSNR (FBP from CNN Sino): {avg_psnr_fbp_cnn:.4f}")
    print(f"平均 SSIM (FBP from CNN Sino): {avg_ssim_fbp_cnn:.4f}")
    print(f"平均 PSNR (GAN Output): {avg_psnr_gan:.4f}")
    print(f"平均 SSIM (GAN Output): {avg_ssim_gan:.4f}")

    return avg_psnr_gan, avg_ssim_gan


def plot_history(history, title, filename):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(10, 5))
    epochs = history.get('epoch')
    if epochs is None:
        print(f"错误：用于绘制'{title}'的历史数据中缺少'epoch'键。")
        plt.close()
        return

    plotted_something = False
    for key, values in history.items():
        if key != 'epoch':
            if len(epochs) == len(values):
                plt.plot(epochs, values, label=key.replace('_', ' ').title())
                plotted_something = True
            else:
                print(f"警告：在图'{title}'中'{key}'的长度不匹配。{len(epochs)}个epoch对应{len(values)}个值。")
    if not plotted_something:
        print(f"警告：没有有效数据可用于绘制'{title}'。")
        plt.close()
        return

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(config.OUTPUT_DIR, filename)
    try:
        plt.savefig(save_path)
        print(f"已保存绘图: {save_path}")
    except Exception as e:
        print(f"保存绘图'{save_path}'时出错: {e}")
    finally:
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用CNN+GAN的稀疏视图CT重建")
    parser.add_argument('--mode', type=str, required=True, choices=['train_cnn', 'train_gan', 'test', 'full_train'],
                        help="操作模式")
    parser.add_argument('--cnn_weights', type=str, default=None, help="预训练ProjectionCNN权重的路径")
    parser.add_argument('--gen_weights', type=str, default=None, help="预训练生成器权重的路径")
    parser.add_argument('--disc_weights', type=str, default=None, help="预训练判别器权重的路径")
    parser.add_argument('--epochs_cnn', type=int, default=config.EPOCHS_CNN, help="CNN训练的Epoch数")
    parser.add_argument('--epochs_gan', type=int, default=config.EPOCHS_GAN, help="GAN训练的Epoch数")
    parser.add_argument('--pretrain_mse', type=int, default=config.PRETRAIN_GAN_MSE_EPOCHS,
                        help="用MSE预训练G的Epoch数")
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help="批量大小")
    parser.add_argument('--num_workers', type=int, default=config.NUM_WORKERS, help="DataLoader的worker数量")
    parser.add_argument('--lambda_adv', type=float, default=config.LAMBDA_ADV, help="生成器中对抗性损失的权重")
    parser.add_argument('--lr_g', type=float, default=config.LEARNING_RATE_G, help="生成器的学习率")
    parser.add_argument('--lr_d', type=float, default=config.LEARNING_RATE_D, help="判别器的学习率")

    args = parser.parse_args()

    config.BATCH_SIZE = args.batch_size
    config.EPOCHS_CNN = args.epochs_cnn
    config.EPOCHS_GAN = args.epochs_gan
    config.PRETRAIN_GAN_MSE_EPOCHS = args.pretrain_mse
    config.NUM_WORKERS = args.num_workers
    config.LAMBDA_ADV = args.lambda_adv
    config.LEARNING_RATE_G = args.lr_g
    config.LEARNING_RATE_D = args.lr_d

    print("\n--- 生效配置 ---")
    print(f"设备: {config.DEVICE}")
    print(f"批量大小: {config.BATCH_SIZE}")
    print(f"CNN Epochs: {config.EPOCHS_CNN}")
    print(f"GAN Epochs: {config.EPOCHS_GAN}")
    print(f"生成器MSE预训练 Epochs: {config.PRETRAIN_GAN_MSE_EPOCHS}")
    print(f"Num Workers: {config.NUM_WORKERS}")
    print(
        f"Lambda ADV: {config.LAMBDA_ADV}, Lambda MSE: {config.LAMBDA_MSE}, Lambda Content: {config.LAMBDA_CONTENT}, Lambda TV: {config.LAMBDA_TV}")
    print(f"LR CNN: {config.LEARNING_RATE_CNN}, LR G: {config.LEARNING_RATE_G}, LR D: {config.LEARNING_RATE_D}")
    print(f"标签平滑 Real: {config.LABEL_SMOOTHING_REAL}, Fake: {config.LABEL_SMOOTHING_FAKE}")
    print(f"D 更新频率: {config.D_UPDATE_FREQ}, 梯度裁剪: {config.GRAD_CLIP_VALUE}")
    print(f"输出目录: {config.OUTPUT_DIR}")
    print(f"使用的稀疏角度: 从 {config.NUM_ANGLES_FULL} 个总潜在视图中选取 {len(config.SPARSE_VIEW_ANGLES)} 个视图。")
    print("-----------------------------\n")

    device = config.DEVICE
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("加载数据集中...")
    train_path = os.path.join(config.DATA_DIR, config.TRAIN_FILE)
    test_path = os.path.join(config.DATA_DIR, config.TEST_FILE)

    train_dataset = CTDataset(npz_path=train_path, sparse_angles_idx=config.SPARSE_VIEW_ANGLES, is_train=True,
                              normalize_img=config.normalize_img, normalize_sino=config.normalize_sino)
    test_dataset = CTDataset(npz_path=test_path, sparse_angles_idx=config.SPARSE_VIEW_ANGLES, is_train=False,
                             normalize_img=config.normalize_img, normalize_sino=config.normalize_sino)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS,
                              pin_memory=(device.type == 'cuda'), drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS,
                             pin_memory=(device.type == 'cuda'), drop_last=False)
    print("数据集加载完成。")

    print("初始化模型中...")
    projection_cnn = ProjectionCNN(in_channels=config.IMG_CHANNELS).to(device)  # 假设sinogram通道为1
    generator = Generator(
        in_channels=config.IMG_CHANNELS_GEN, out_channels=config.IMG_CHANNELS_GEN,
        features=config.GEN_FEATURES, num_blocks=config.GEN_NUM_BLOCKS,
        bottleneck_expansion=config.GEN_BOTTLENECK_EXPANSION,
        use_eca=config.USE_ECA_IN_GENERATOR, eca_k_size=config.ECA_KERNEL_SIZE
    ).to(device)
    discriminator = Discriminator(
        in_channels=config.IMG_CHANNELS_DISC, img_size=config.IMAGE_SIZE,
        features_start=config.DISC_FEATURES_START
    ).to(device)
    print("模型初始化完成。")

    if args.mode == 'train_cnn' or args.mode == 'full_train':
        if args.cnn_weights and os.path.exists(args.cnn_weights) and args.mode != 'full_train':
            print(f"加载现有CNN权重: {args.cnn_weights}")
            projection_cnn.load_state_dict(torch.load(args.cnn_weights, map_location=device))
        optimizer_cnn = optim.Adam(projection_cnn.parameters(), lr=config.LEARNING_RATE_CNN,
                                   betas=(config.ADAM_BETA1, config.ADAM_BETA2))
        criterion_cnn = nn.MSELoss().to(device)  # 或 L1Loss
        cnn_history = train_cnn(projection_cnn, train_loader, optimizer_cnn, criterion_cnn, device, config.EPOCHS_CNN)
        plot_history({'loss': cnn_history['loss'], 'epoch': cnn_history['epoch']}, "Projection CNN Training Loss",
                     "cnn_loss_history.png")
        num_sparse_views = config.num_sparse_views if hasattr(config, "num_sparse_views") else len(config.SPARSE_VIEW_ANGLES)
        
        final_cnn_path = os.path.join(
            config.OUTPUT_DIR,
            f'projection_cnn_final_{num_sparse_views}views.pth'
        )
        
        torch.save(projection_cnn.state_dict(), final_cnn_path)
        print(f"已保存最终CNN模型到 {final_cnn_path}")

        args.cnn_weights = final_cnn_path

    if args.mode == 'train_gan' or args.mode == 'full_train':
        if not args.cnn_weights or not os.path.exists(args.cnn_weights):
            raise ValueError("train_gan/full_train 模式需要CNN权重 (--cnn_weights)。")
        print(f"为GAN训练加载CNN权重: {args.cnn_weights}")
        projection_cnn.load_state_dict(torch.load(args.cnn_weights, map_location=device))

        if args.gen_weights and os.path.exists(args.gen_weights):
            print(f"加载现有生成器权重: {args.gen_weights}")
            generator.load_state_dict(torch.load(args.gen_weights, map_location=device))
        if args.disc_weights and os.path.exists(args.disc_weights):
            print(f"加载现有判别器权重: {args.disc_weights}")
            discriminator.load_state_dict(torch.load(args.disc_weights, map_location=device))

        optimizer_g = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE_G,
                                 betas=(config.ADAM_BETA1, config.ADAM_BETA2))
        optimizer_d = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE_D,
                                 betas=(config.ADAM_BETA1, config.ADAM_BETA2))

        gan_history = train_gan(projection_cnn, generator, discriminator, train_loader, optimizer_g, optimizer_d,
                                device, config.EPOCHS_GAN, config.PRETRAIN_GAN_MSE_EPOCHS)
        plot_history({'Generator Loss': gan_history['g_loss'], 'Discriminator Loss': gan_history['d_loss'],
                      'epoch': gan_history['epoch']}, "GAN Training Loss", "gan_loss_history.png")
        plot_history({'PSNR': gan_history['psnr'], 'SSIM': gan_history['ssim'], 'epoch': gan_history['epoch']},
                     "GAN Training Metrics", "gan_metrics_history.png")

        # -------------------------
        #  Final Save (Generator + Discriminator)
        # -------------------------
        
        # 获取稀疏视图数量（优先使用 config.num_sparse_views，没有就用 SPARSE_VIEW_ANGLES 的长度）
        num_sparse_views = (
            config.num_sparse_views
            if hasattr(config, "num_sparse_views")
            else len(config.SPARSE_VIEW_ANGLES)
        )
        
        # 生成文件名
        final_g_path = os.path.join(
            config.OUTPUT_DIR,
            f"generator_final_{num_sparse_views}views.pth"
        )
        final_d_path = os.path.join(
            config.OUTPUT_DIR,
            f"discriminator_final_{num_sparse_views}views.pth"
        )
        
        # 保存模型
        torch.save(generator.state_dict(), final_g_path)
        torch.save(discriminator.state_dict(), final_d_path)
        
        print(f"已保存最终生成器模型到 {final_g_path}")
        print(f"已保存最终判别器模型到 {final_d_path}")

        args.gen_weights = final_g_path

    if args.mode == 'test':
        if not args.cnn_weights or not os.path.exists(args.cnn_weights):
            raise ValueError("测试模式需要CNN权重 (--cnn_weights)。")
        if not args.gen_weights or not os.path.exists(args.gen_weights):
            raise ValueError("测试模式需要生成器权重 (--gen_weights)。")

        print(f"为测试加载CNN权重: {args.cnn_weights}")
        projection_cnn.load_state_dict(torch.load(args.cnn_weights, map_location=device))
        print(f"为测试加载生成器权重: {args.gen_weights}")
        generator.load_state_dict(torch.load(args.gen_weights, map_location=device))
        test(projection_cnn, generator, test_loader, device)

    print("\n处理完成。")
