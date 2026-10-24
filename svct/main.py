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
from scipy.interpolate import interp1d  # <<< 保留：线性插值
import swanlab
# import gc

def print_memory_usage():
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
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
    "PRETRAIN_GAN_MSE_EPOCHS": 100,
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
    calculate_psnr, calculate_ssim
)

# ==================== 统一的 checkpoint 目录函数 ====================

def get_checkpoint_dir():
    """
    根据稀疏视图数量生成子文件夹路径:
    {config.OUTPUT_DIR}/{num_sparse_views}views
    """
    num_sparse_views = (
        config.num_sparse_views
        if hasattr(config, "num_sparse_views")
        else len(config.SPARSE_VIEW_ANGLES)
    )
    ckpt_dir = os.path.join(config.OUTPUT_DIR, f"{num_sparse_views}views")
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir, num_sparse_views

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
    xp = np.arange(full_num_angles)  # 定义插值x坐标（所有角度）

    for b_idx in range(batch_size):
        known_projection_values = sparse_sino_np[b_idx, 0, sparse_angles_indices, :]
        interpolated_sino_single_batch = np.zeros((full_num_angles, num_detectors), dtype=np.float32)

        for det_idx in range(num_detectors):
            fp_detector = known_projection_values[:, det_idx]
            interp_func = interp1d(
                sparse_angles_indices,
                fp_detector,
                kind='linear',
                bounds_error=False,
                fill_value="extrapolate"
            )
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

        # ---- 保存中间 CNN checkpoint（放在视图子文件夹下）----
        if (epoch + 1) % config.SAVE_INTERVAL == 0 or (epoch + 1) == epochs:
            ckpt_dir, num_sparse_views = get_checkpoint_dir()
            save_path = os.path.join(ckpt_dir, f'projection_cnn_epoch_{epoch + 1}.pth')
            torch.save(cnn_model.state_dict(), save_path)
            print(f"已保存CNN模型检查点到 {save_path}")

    print("--- 投影CNN训练完成 ---")
    return history

def precompute_fbp_npz(cnn_model, dataset, device, save_path):
    """
    使用训练好的 projection_cnn，把给定 dataset 中的稀疏 sinogram 全部
    映射成 FBP 图像，并连同 ct_label 一起保存到 npz。

    注意：
      - 使用的是和 train_cnn 相同的 CTDataset（包含 sparse_sinogram & ct_label）
      - 不在这里做 backprop，只 forward + FBP + 归一化
    """
    cnn_model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        drop_last=False
    )

    fbp_list = []
    ct_list = []

    print(f"\n--- 开始预计算 FBP 并写入 {save_path} ---")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Precompute FBP", leave=True):
            sparse_sinogram = batch["sparse_sinogram"].to(device)
            ct_label = batch["ct_label"].to(device)

            # 1. CNN 插值 sinogram
            interpolated_sino_cnn = cnn_model(sparse_sinogram)

            # 2. FBP 重建
            fbp_from_cnn_output_raw = fbp_reconstruct(
                interpolated_sino_cnn, config.ANGLES_RAD,
                output_size=config.IMAGE_SIZE
            )

            # 3. 归一化（完全复用你原来 train_gan 里的归一化逻辑）
            if fbp_from_cnn_output_raw.numel() > 0:
                fbp_min = fbp_from_cnn_output_raw.min()
                fbp_max = fbp_from_cnn_output_raw.max()
                if fbp_max > fbp_min:
                    fbp_norm = (fbp_from_cnn_output_raw - fbp_min) / (fbp_max - fbp_min)
                    if config.normalize_img == (-1, 1):
                        fbp_norm = fbp_norm * 2.0 - 1.0
                else:
                    fbp_norm = torch.full_like(
                        fbp_from_cnn_output_raw,
                        float(config.normalize_img[0])
                    )
            else:
                fbp_norm = torch.full_like(
                    fbp_from_cnn_output_raw,
                    float(config.normalize_img[0])
                )

            # 4. 转成 numpy、放到列表中（注意要 .cpu()）
            fbp_list.append(fbp_norm.cpu().numpy().astype(np.float32))
            ct_list.append(ct_label.cpu().numpy().astype(np.float32))

            # 防止显存/内存堆积
            del sparse_sinogram, ct_label, interpolated_sino_cnn, fbp_from_cnn_output_raw, fbp_norm
            # gc.collect()
            # torch.cuda.empty_cache()

    # 5. 拼接成大数组
    fbp_array = np.concatenate(fbp_list, axis=0)   # (N, 1, H, W)
    ct_array  = np.concatenate(ct_list, axis=0)    # (N, 1, H, W)
    print(f"预计算 FBP 完成，共 {fbp_array.shape[0]} 张。开始写入 npz ...")

    np.savez_compressed(
        save_path,
        fbp=fbp_array,
        ct_label=ct_array
    )
    print(f"已保存 FBP npz 数据集到 {save_path}")

def precompute_fbp_npz_test(cnn_model, dataset, device, save_path):
    """
    用于 test：预计算三种图像
      - fbp_li   : 线性插值得到的 sinogram，经 FBP + 归一化
      - fbp_cnn  : Projection CNN 输出 sinogram，经 FBP + 归一化
      - ct_label : 真值 CT

    这三个就是你 test() 里的三个分支：
      1) FBP_LI   baseline
      2) FBP_CNN  baseline
      3) GAN( FBP_CNN )
    """
    cnn_model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        drop_last=False
    )

    fbp_li_list = []
    fbp_cnn_list = []
    ct_list = []

    print(f"\n--- [TEST] 开始预计算 FBP(LI & CNN) 并写入 {save_path} ---")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Precompute TEST FBP (LI & CNN)", leave=True):
            sparse_sinogram = batch["sparse_sinogram"].to(device)  # 稀疏 sinogram
            ct_label = batch["ct_label"].to(device)

            # ===== ① CNN → FBP =====
            interpolated_sino_cnn = cnn_model(sparse_sinogram)
            fbp_cnn_raw = fbp_reconstruct(
                interpolated_sino_cnn, config.ANGLES_RAD,
                output_size=config.IMAGE_SIZE
            )
            # 归一化（与训练保持一致）
            if fbp_cnn_raw.numel() > 0:
                fmin = fbp_cnn_raw.min()
                fmax = fbp_cnn_raw.max()
                if fmax > fmin:
                    fbp_cnn_norm = (fbp_cnn_raw - fmin) / (fmax - fmin)
                    if config.normalize_img == (-1, 1):
                        fbp_cnn_norm = fbp_cnn_norm * 2.0 - 1.0
                else:
                    fbp_cnn_norm = torch.full_like(
                        fbp_cnn_raw,
                        float(config.normalize_img[0])
                    )
            else:
                fbp_cnn_norm = torch.full_like(
                    fbp_cnn_raw,
                    float(config.normalize_img[0])
                )

            # ===== ② 线性插值 → FBP =====
            linearly_interpolated_sino = linear_interpolate_sinogram_torch(
                sparse_sinogram,
                config.SPARSE_VIEW_ANGLES,
                config.NUM_ANGLES_FULL,
                device
            )
            fbp_li_raw = fbp_reconstruct(
                linearly_interpolated_sino, config.ANGLES_RAD,
                output_size=config.IMAGE_SIZE
            )
            if fbp_li_raw.numel() > 0:
                fmin_li = fbp_li_raw.min()
                fmax_li = fbp_li_raw.max()
                if fmax_li > fmin_li:
                    fbp_li_norm = (fbp_li_raw - fmin_li) / (fmax_li - fmin_li)
                    if config.normalize_img == (-1, 1):
                        fbp_li_norm = fbp_li_norm * 2.0 - 1.0
                else:
                    fbp_li_norm = torch.full_like(
                        fbp_li_raw,
                        float(config.normalize_img[0])
                    )
            else:
                fbp_li_norm = torch.full_like(
                    fbp_li_raw,
                    float(config.normalize_img[0])
                )

            # ===== 收集 =====
            fbp_cnn_list.append(fbp_cnn_norm.cpu().numpy().astype(np.float32))
            fbp_li_list.append(fbp_li_norm.cpu().numpy().astype(np.float32))
            ct_list.append(ct_label.cpu().numpy().astype(np.float32))

            # # 内存清理
            # del sparse_sinogram, ct_label
            # del interpolated_sino_cnn, fbp_cnn_raw, fbp_cnn_norm
            # del linearly_interpolated_sino, fbp_li_raw, fbp_li_norm
            # gc.collect()
            # torch.cuda.empty_cache()

    fbp_cnn_array = np.concatenate(fbp_cnn_list, axis=0)  # (N, 1, H, W)
    fbp_li_array  = np.concatenate(fbp_li_list,  axis=0)  # (N, 1, H, W)
    ct_array      = np.concatenate(ct_list,      axis=0)  # (N, 1, H, W)

    print(f"[TEST] 预计算完成，共 {fbp_cnn_array.shape[0]} 张，开始写入 npz...")

    np.savez_compressed(
        save_path,
        fbp_li=fbp_li_array,
        fbp_cnn=fbp_cnn_array,
        ct_label=ct_array
    )
    print(f"[TEST] 已保存 FBP test npz 数据集到 {save_path}")



def train_gan(generator, discriminator, train_loader, optim_g, optim_d,
              device, epochs, pretrain_mse_epochs):
    """
    使用预计算好的 FBP 图像训练 GAN。
    train_loader 应该来自 FBPCTDataset，batch 中包含:
        - batch["fbp_image"]: 已经过 CNN + FBP + 归一化 的图像
        - batch["ct_label"] : 真实 CT 图像
    """
    print("\n--- 开始GAN训练（使用预计算 FBP） ---")
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

    # =========== 生成器 MSE 预训练 ===========
    if pretrain_mse_epochs > 0:
        print(f"\n--- 使用MSE预训练生成器 {pretrain_mse_epochs} 个epoch ---")
        optim_g_pretrain = optim.Adam(
            generator.parameters(),
            lr=config.LEARNING_RATE_G,
            betas=(config.ADAM_BETA1, config.ADAM_BETA2)
        )

        for epoch in range(pretrain_mse_epochs):
            epoch_g_loss_pre = 0.0
            progress_bar = tqdm(
                train_loader,
                desc=f"预训练 Epoch {epoch + 1}/{pretrain_mse_epochs}",
                leave=True
            )
            for batch in progress_bar:
                fbp_image = batch["fbp_image"].to(device)   # 直接用预计算 FBP
                ct_label  = batch["ct_label"].to(device)

                optim_g_pretrain.zero_grad()
                generated_image = generator(fbp_image)
                loss_g_mse = criterion_pixel(generated_image, ct_label)
                loss_g_mse.backward()
                optim_g_pretrain.step()

                epoch_g_loss_pre += loss_g_mse.item()
                progress_bar.set_postfix(mse_loss=f"{loss_g_mse.item():.4f}")

                del fbp_image, ct_label, generated_image
                # gc.collect()
                # torch.cuda.empty_cache()

            avg_epoch_g_loss_pre = epoch_g_loss_pre / len(train_loader)
            print(f"预训练 Epoch {epoch + 1}/{pretrain_mse_epochs} - 生成器 MSE Loss: {avg_epoch_g_loss_pre:.6f}")
            swanlab.log({"生成器 MSE Loss": avg_epoch_g_loss_pre})

            print_memory_usage()

            # ======= 预训练阶段保存生成器权重 =======
            if (epoch + 1) % 20 == 0 or (epoch + 1) == pretrain_mse_epochs:
                ckpt_dir, num_sparse_views = get_checkpoint_dir()
                pre_g_path = os.path.join(
                    ckpt_dir,
                    f"generator_pretrain_epoch_{epoch + 1}.pth"
                )
                torch.save(generator.state_dict(), pre_g_path)
                print(f"[预训练] 已保存生成器检查点到 {pre_g_path}")
        print("--- 生成器预训练完成 ---")

    # =========== 对抗训练 ===========
    print("\n--- 开始对抗性训练 ---")
    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f"GAN Epoch {epoch + 1}/{epochs}", leave=True)

        for i, batch in enumerate(progress_bar):
            fbp_image = batch["fbp_image"].to(device)  # 预计算 FBP
            real_image = batch["ct_label"].to(device)

            current_batch_size = real_image.size(0)
            current_real_label = (
                real_label_base[:current_batch_size]
                if current_batch_size != config.BATCH_SIZE
                else real_label_base
            )
            current_fake_label = (
                fake_label_base[:current_batch_size]
                if current_batch_size != config.BATCH_SIZE
                else fake_label_base
            )

            # --- 训练判别器 ---
            optim_d.zero_grad()
            fake_image = generator(fbp_image)

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
            loss_g_val = 0.0
            if (i + 1) % config.D_UPDATE_FREQ == 0:
                optim_g.zero_grad()
                pred_fake_g = discriminator(fake_image)
                loss_g_adv = criterion_gan(pred_fake_g, current_real_label)
                loss_g_pixel = criterion_pixel(fake_image, real_image)
                loss_g_content = criterion_content(fake_image, real_image)
                loss_g_tv = total_variation_loss(fake_image)

                loss_g = (
                    config.LAMBDA_MSE * loss_g_pixel +
                    config.LAMBDA_ADV * loss_g_adv +
                    config.LAMBDA_CONTENT * loss_g_content +
                    config.LAMBDA_TV * loss_g_tv
                )
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
                G_loss=f"{loss_g_val:.4f}",
                D_loss=f"{loss_d.item():.4f}",
                PSNR=f"{psnr_val:.2f}",
                SSIM=f"{ssim_val:.3f}"
            )

            del fbp_image, real_image, fake_image, pred_real, pred_fake
            # gc.collect()
            # torch.cuda.empty_cache()

        avg_epoch_g_loss = (
            epoch_g_loss / (len(train_loader) / config.D_UPDATE_FREQ)
            if config.D_UPDATE_FREQ > 0
            else epoch_g_loss / len(train_loader)
        )
        avg_epoch_d_loss = epoch_d_loss / len(train_loader)
        avg_epoch_psnr = epoch_psnr / len(train_loader)
        avg_epoch_ssim = epoch_ssim / len(train_loader)
        epoch_time = time.time() - start_time

        print(
            f"GAN Epoch {epoch + 1}/{epochs} - Time: {epoch_time:.2f}s\n"
            f"  Avg G Loss: {avg_epoch_g_loss:.6f}, Avg D Loss: {avg_epoch_d_loss:.6f}\n"
            f"  Avg PSNR: {avg_epoch_psnr:.4f}, Avg SSIM: {avg_epoch_ssim:.4f}"
        )
        swanlab.log({
            "Avg G Loss": avg_epoch_g_loss,
            "Avg D Loss": avg_epoch_d_loss,
            "Avg PSNR": avg_epoch_psnr,
            "Avg SSIM": avg_epoch_ssim
        })

        history['epoch'].append(epoch + 1)
        history['g_loss'].append(avg_epoch_g_loss)
        history['d_loss'].append(avg_epoch_d_loss)
        history['psnr'].append(avg_epoch_psnr)
        history['ssim'].append(avg_epoch_ssim)

        # ---- 保存中间 GAN checkpoint（放在视图子文件夹下）----
        if (epoch + 1) % config.SAVE_INTERVAL == 0 or (epoch + 1) == epochs:
            ckpt_dir, num_sparse_views = get_checkpoint_dir()
            save_path_g = os.path.join(ckpt_dir, f'generator_epoch_{epoch + 1}.pth')
            save_path_d = os.path.join(ckpt_dir, f'discriminator_epoch_{epoch + 1}.pth')
            torch.save(generator.state_dict(), save_path_g)
            torch.save(discriminator.state_dict(), save_path_d)
            print(f"已保存GAN模型检查点到 {ckpt_dir}")

    print_memory_usage()
    # gc.collect()
    # torch.cuda.empty_cache()

    print("--- GAN训练完成 ---")
    swanlab.finish()
    return history


def test(generator, test_loader, device):
    """
    使用预计算 FBP npz 进行测试。
    前两个测试项：
        - PSNR_FBP_LI / SSIM_FBP_LI   : 线性插值得到的 sinogram，经 FBP 的结果
    后两个基线：
        - PSNR_FBP_CNN / SSIM_FBP_CNN : Projection CNN 输出 sinogram，经 FBP 的结果
    最后：
        - PSNR_GAN / SSIM_GAN         : 生成器对 CNN-FBP 的增强结果
    """
    print("\n--- 开始测试阶段（使用预计算 FBP：LI & CNN） ---")
    generator.eval()

    total_psnr_fbp_li = 0.0
    total_ssim_fbp_li = 0.0
    total_psnr_fbp_cnn = 0.0
    total_ssim_fbp_cnn = 0.0
    total_psnr_gan = 0.0
    total_ssim_gan = 0.0

    start_time = time.time()
    progress_bar = tqdm(test_loader, desc="测试中", leave=False)

    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            # 从 FBP npz 读取三种图像
            fbp_li   = batch['fbp_li'].to(device)      # 线性插值 → FBP
            fbp_cnn  = batch['fbp_cnn'].to(device)     # CNN 插值 → FBP
            ct_label = batch['ct_label'].to(device)    # 真值 CT

            # GAN 的输入仍然是 CNN-FBP（与训练保持一致）
            gan_output_image = generator(fbp_cnn)

            data_range = 2.0 if config.normalize_img == (-1, 1) else 1.0
            mch = False if ct_label.ndim == 4 and ct_label.size(1) == 1 else True

            # -------- ① 线性插值 FBP baseline（前两个测试项） --------
            psnr_fbp_li_val = calculate_psnr(ct_label, fbp_li, data_range=data_range)
            ssim_fbp_li_val = calculate_ssim(ct_label, fbp_li, data_range=data_range, multichannel=mch)
            total_psnr_fbp_li += psnr_fbp_li_val
            total_ssim_fbp_li += ssim_fbp_li_val

            # -------- ② CNN-FBP baseline --------
            psnr_fbp_cnn_val = calculate_psnr(ct_label, fbp_cnn, data_range=data_range)
            ssim_fbp_cnn_val = calculate_ssim(ct_label, fbp_cnn, data_range=data_range, multichannel=mch)
            total_psnr_fbp_cnn += psnr_fbp_cnn_val
            total_ssim_fbp_cnn += ssim_fbp_cnn_val

            # -------- ③ GAN 输出 --------
            psnr_gan_val = calculate_psnr(ct_label, gan_output_image, data_range=data_range)
            ssim_gan_val = calculate_ssim(ct_label, gan_output_image, data_range=data_range, multichannel=mch)
            total_psnr_gan += psnr_gan_val
            total_ssim_gan += ssim_gan_val

            # 与你原来类似的打印顺序
            progress_bar.set_postfix(
                PSNR_GAN=f"{psnr_gan_val:.2f}",
                SSIM_GAN=f"{ssim_gan_val:.3f}",
                PSNR_FBP_CNN=f"{psnr_fbp_cnn_val:.2f}",
                PSNR_FBP_LI=f"{psnr_fbp_li_val:.2f}"
            )

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
    print(f"平均 PSNR (FBP from Linearly Interpolated Sino): {avg_psnr_fbp_li:.4f}")
    print(f"平均 SSIM (FBP from Linearly Interpolated Sino): {avg_ssim_fbp_li:.4f}")
    print(f"平均 PSNR (FBP from CNN Sino): {avg_psnr_fbp_cnn:.4f}")
    print(f"平均 SSIM (FBP from CNN Sino): {avg_ssim_fbp_cnn:.4f}")
    print(f"平均 PSNR (GAN Output): {avg_psnr_gan:.4f}")
    print(f"平均 SSIM (GAN Output): {avg_ssim_gan:.4f}")

    return avg_psnr_gan, avg_ssim_gan



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用CNN+GAN的稀疏视图CT重建")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train_cnn', 'train_gan', 'test', 'full_train', 'precompute_fbp'],
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
    parser.add_argument('--lambda_adv', type=float, default=config.LAMBDA_ADV,
                        help="生成器中对抗性损失的权重")
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
        f"Lambda ADV: {config.LAMBDA_ADV}, Lambda MSE: {config.LAMBDA_MSE}, "
        f"Lambda Content: {config.LAMBDA_CONTENT}, Lambda TV: {config.LAMBDA_TV}"
    )
    print(f"LR CNN: {config.LEARNING_RATE_CNN}, LR G: {config.LEARNING_RATE_G}, LR D: {config.LEARNING_RATE_D}")
    print(f"标签平滑 Real: {config.LABEL_SMOOTHING_REAL}, Fake: {config.LABEL_SMOOTHING_FAKE}")
    print(f"D 更新频率: {config.D_UPDATE_FREQ}, 梯度裁剪: {config.GRAD_CLIP_VALUE}")
    print(f"输出目录: {config.OUTPUT_DIR}")
    print(
        f"使用的稀疏角度: 从 {config.NUM_ANGLES_FULL} 个总潜在视图中选取 "
        f"{len(config.SPARSE_VIEW_ANGLES)} 个视图。"
    )
    print("-----------------------------\n")

    device = config.DEVICE
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("加载数据集中...")
    train_path = os.path.join(config.DATA_DIR, config.TRAIN_FILE)
    test_path = os.path.join(config.DATA_DIR, config.TEST_FILE)

    # 你希望 FBP 数据集也存到 /root/autodl-fs/dataset/ 目录
    train_fbp_path = os.path.join(config.DATA_DIR, config.TRAIN_FBP_FILE)
    test_fbp_path  = os.path.join(config.DATA_DIR, config.TEST_FBP_FILE)
    
    train_dataset = CTDataset(
        npz_path=train_path,
        sparse_angles_idx=config.SPARSE_VIEW_ANGLES,
        is_train=True,
        normalize_img=config.normalize_img,
        normalize_sino=config.normalize_sino
    )
    test_dataset = CTDataset(
        npz_path=test_path,
        sparse_angles_idx=config.SPARSE_VIEW_ANGLES,
        is_train=False,
        normalize_img=config.normalize_img,
        normalize_sino=config.normalize_sino
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == 'cuda'),
        drop_last=False
    )
    print("数据集加载完成。")

    print("初始化模型中...")
    projection_cnn = ProjectionCNN(in_channels=config.IMG_CHANNELS).to(device)
    generator = Generator(
        in_channels=config.IMG_CHANNELS_GEN,
        out_channels=config.IMG_CHANNELS_GEN,
        features=config.GEN_FEATURES,
        num_blocks=config.GEN_NUM_BLOCKS,
        bottleneck_expansion=config.GEN_BOTTLENECK_EXPANSION,
        use_eca=config.USE_ECA_IN_GENERATOR,
        eca_k_size=config.ECA_KERNEL_SIZE
    ).to(device)
    discriminator = Discriminator(
        in_channels=config.IMG_CHANNELS_DISC,
        img_size=config.IMAGE_SIZE,
        features_start=config.DISC_FEATURES_START
    ).to(device)
    print("模型初始化完成。")

    # ========== CNN 训练 ==========
    if args.mode == 'train_cnn' or args.mode == 'full_train':
        if args.cnn_weights and os.path.exists(args.cnn_weights) and args.mode != 'full_train':
            print(f"加载现有CNN权重: {args.cnn_weights}")
            projection_cnn.load_state_dict(torch.load(args.cnn_weights, map_location=device))

        optimizer_cnn = optim.Adam(
            projection_cnn.parameters(),
            lr=config.LEARNING_RATE_CNN,
            betas=(config.ADAM_BETA1, config.ADAM_BETA2)
        )
        criterion_cnn = nn.MSELoss().to(device)
        cnn_history = train_cnn(
            projection_cnn, train_loader, optimizer_cnn, criterion_cnn, device, config.EPOCHS_CNN
        )

        # 最终 CNN checkpoint 保存到视图子文件夹
        ckpt_dir, num_sparse_views = get_checkpoint_dir()
        final_cnn_path = os.path.join(
            ckpt_dir,
            f'projection_cnn_final_{num_sparse_views}views.pth'
        )
        torch.save(projection_cnn.state_dict(), final_cnn_path)
        print(f"已保存最终CNN模型到 {final_cnn_path}")

        args.cnn_weights = final_cnn_path

    if args.mode == 'precompute_fbp':
        if not args.cnn_weights or not os.path.exists(args.cnn_weights):
            raise ValueError("precompute_fbp 模式需要 CNN 权重 (--cnn_weights)。")
    
        print(f"加载 CNN 权重: {args.cnn_weights}")
        projection_cnn.load_state_dict(torch.load(args.cnn_weights, map_location=device))
    
        # 生成 train FBP
        train_fbp_path = os.path.join(config.DATA_DIR, config.TRAIN_FBP_FILE)
        if not os.path.exists(train_fbp_path):
            print(f"生成训练集 FBP 数据集: {train_fbp_path}")
            precompute_fbp_npz(projection_cnn, train_dataset, device, train_fbp_path)
        else:
            print(f"训练集 FBP 已存在: {train_fbp_path}")
    
        # 生成 test FBP
        test_fbp_path = os.path.join(config.DATA_DIR, config.TEST_FBP_FILE)
        if not os.path.exists(test_fbp_path):
            print(f"生成测试集 FBP 数据集: {test_fbp_path}")
            precompute_fbp_npz_test(projection_cnn, test_dataset, device, test_fbp_path)
        else:
            print(f"测试集 FBP 已存在: {test_fbp_path}")
    
        print("FBP 数据集生成完成。退出。")
        exit()


    # ========== GAN 训练 ==========
    if args.mode == 'train_gan' or args.mode == 'full_train':
        if not args.cnn_weights or not os.path.exists(args.cnn_weights):
            raise ValueError("train_gan/full_train 模式需要CNN权重 (--cnn_weights)。")
        print(f"为GAN训练加载CNN权重: {args.cnn_weights}")
        projection_cnn.load_state_dict(torch.load(args.cnn_weights, map_location=device))

        # 1) 如果 FBP 训练集 npz 不存在，先预计算
        if not os.path.exists(train_fbp_path):
            print(f"未找到预计算的 FBP 训练集 {train_fbp_path}，开始生成...")
            precompute_fbp_npz(projection_cnn, train_dataset, device, train_fbp_path)
        else:
            print(f"检测到已有 FBP 训练集: {train_fbp_path}，直接加载。")

        # 2) 使用 FBPCTDataset 创建专用 GAN 训练集和 DataLoader
        from dataset import FBPCTDataset  # 文件顶部也可以统一 import
        gan_train_dataset = FBPCTDataset(
            npz_path=train_fbp_path,
            is_train=True,
            normalize_img=config.normalize_img
        )
        gan_train_loader = DataLoader(
            gan_train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            drop_last=True
        )

        # 3) 按需加载已有 G / D 权重
        if args.gen_weights and os.path.exists(args.gen_weights):
            print(f"加载现有生成器权重: {args.gen_weights}")
            generator.load_state_dict(torch.load(args.gen_weights, map_location=device))
        if args.disc_weights and os.path.exists(args.disc_weights):
            print(f"加载现有判别器权重: {args.disc_weights}")
            discriminator.load_state_dict(torch.load(args.disc_weights, map_location=device))

        optimizer_g = optim.Adam(
            generator.parameters(),
            lr=config.LEARNING_RATE_G,
            betas=(config.ADAM_BETA1, config.ADAM_BETA2)
        )
        optimizer_d = optim.Adam(
            discriminator.parameters(),
            lr=config.LEARNING_RATE_D,
            betas=(config.ADAM_BETA1, config.ADAM_BETA2)
        )

        # 4) 注意：这里调用的是“新签名”的 train_gan（不再传 cnn_model）
        gan_history = train_gan(
            generator, discriminator,
            gan_train_loader, optimizer_g, optimizer_d,
            device, config.EPOCHS_GAN, config.PRETRAIN_GAN_MSE_EPOCHS
        )

        # 最终 GAN checkpoint 保存到视图子文件夹
        ckpt_dir, num_sparse_views = get_checkpoint_dir()
        
        # 根据是否启用 ECA 生成后缀，确保不会覆盖
        eca_tag = "eca" if config.USE_ECA_IN_GENERATOR else "noeca"
        
        final_g_path = os.path.join(
            ckpt_dir, f"generator_final_{num_sparse_views}views_{eca_tag}.pth"
        )
        final_d_path = os.path.join(
            ckpt_dir, f"discriminator_final_{num_sparse_views}views_{eca_tag}.pth"
        )
        
        torch.save(generator.state_dict(), final_g_path)
        torch.save(discriminator.state_dict(), final_d_path)
        
        print(f"已保存最终生成器模型到 {final_g_path}")
        print(f"已保存最终判别器模型到 {final_d_path}")


        args.gen_weights = final_g_path

    
   

# ========== 测试 ==========
if args.mode == 'test':
    if not args.cnn_weights or not os.path.exists(args.cnn_weights):
        raise ValueError("测试模式需要CNN权重 (--cnn_weights)。")
    if not args.gen_weights or not os.path.exists(args.gen_weights):
        raise ValueError("测试模式需要生成器权重 (--gen_weights)。")

    print(f"为测试加载CNN权重: {args.cnn_weights}")
    projection_cnn.load_state_dict(torch.load(args.cnn_weights, map_location=device))
    print(f"为测试加载生成器权重: {args.gen_weights}")
    generator.load_state_dict(torch.load(args.gen_weights, map_location=device))

    # 1) 如果 test FBP npz 不存在，先预计算
    if not os.path.exists(test_fbp_path):
        print(f"未找到 FBP test 数据集 {test_fbp_path}，开始预计算...")
        precompute_fbp_npz_test(projection_cnn, test_dataset, device, test_fbp_path)
    else:
        print(f"检测到已有 FBP test 数据集: {test_fbp_path}")

    # 2) 用 FBPCTTestDataset 构造 DataLoader
    from dataset import FBPCTTestDataset
    test_fbp_dataset = FBPCTTestDataset(test_fbp_path)
    test_fbp_loader = DataLoader(
        test_fbp_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        drop_last=False
    )

    # 3) 调用新的 test（不再传 cnn_model）
    test(generator, test_fbp_loader, device)

