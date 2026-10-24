import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import config

class CTDataset(Dataset):
    
    def __init__(self, npz_path, sparse_angles_idx, is_train=True, normalize_img=(-1, 1), normalize_sino=(0, 1)):
    
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")

        print(f"Loading dataset from: {npz_path}")
        data = np.load(npz_path)
        #从 .npz 文件中加载数据
        self.ct_images = data['ct_label']
        self.full_sinograms = data['sin357']

        print(f"  Loaded ct_images shape: {self.ct_images.shape}")
        print(f"  Loaded full_sinograms shape: {self.full_sinograms.shape}")

        # 调整数据形状
        if self.ct_images.ndim == 4 and self.ct_images.shape[-1] == 1:
            self.ct_images = np.transpose(self.ct_images, (0, 3, 1, 2)) # (N, 1, H, W)
        elif self.ct_images.ndim == 3: 
             self.ct_images = np.expand_dims(self.ct_images, axis=1) # (N, 1, H, W)
        else:
             raise ValueError(f"Unexpected ct_images shape: {self.ct_images.shape}")


        # 增加通道维度
        self.full_sinograms = np.expand_dims(self.full_sinograms, axis=1) # (N, 1, Angles, Detectors)

        print(f"  Adjusted ct_images shape (N, C, H, W): {self.ct_images.shape}")
        print(f"  Adjusted full_sinograms shape (N, C, A, D): {self.full_sinograms.shape}")


        self.sparse_angles_idx = sparse_angles_idx
        self.normalize_img = normalize_img
        self.normalize_sino = normalize_sino
        self.is_train = is_train

    def __len__(self):
        return len(self.ct_images)

    def _normalize(self, data, current_min, current_max, target_min, target_max):
        """将数据从 [current_min, current_max] 归一化到 [target_min, target_max]"""
        if current_max == current_min:
             return np.full_like(data, target_min)
        data_normalized = (data - current_min) / (current_max - current_min) # to [0, 1]
        data_scaled = data_normalized * (target_max - target_min) + target_min # to [target_min, target_max]
        return data_scaled

    def __getitem__(self, idx):
        full_sino = self.full_sinograms[idx].astype(np.float32)
        ct_label = self.ct_images[idx].astype(np.float32)

        sparse_sino = np.zeros_like(full_sino)
        sparse_sino[0, self.sparse_angles_idx, :] = full_sino[0, self.sparse_angles_idx, :]


        # Normalization
        sino_min_val = full_sino.min()
        sino_max_val = full_sino.max()
        img_min_val = ct_label.min()
        img_max_val = ct_label.max()

        # Option 2: Use pre-calculated global min/max (uncomment calculation in __init__)
        # sino_min_val = self.sino_min
        # sino_max_val = self.sino_max
        # img_min_val = self.img_min
        # img_max_val = self.img_max

        if self.normalize_sino:
            target_min, target_max = self.normalize_sino
            full_sino = self._normalize(full_sino, sino_min_val, sino_max_val, target_min, target_max)
            sparse_sino = self._normalize(sparse_sino, sino_min_val, sino_max_val, target_min, target_max)


        if self.normalize_img:
             target_min, target_max = self.normalize_img
             ct_label = self._normalize(ct_label, img_min_val, img_max_val, target_min, target_max)


        # Convert to PyTorch tensors
        sparse_sino_tensor = torch.from_numpy(sparse_sino)
        full_sino_tensor = torch.from_numpy(full_sino)
        ct_label_tensor = torch.from_numpy(ct_label)

        return {
            'sparse_sinogram': sparse_sino_tensor,
            'full_sinogram': full_sino_tensor,
            'ct_label': ct_label_tensor
        }

class FBPCTDataset(Dataset):
    """
    专门给 GAN 训练用的数据集：
    从预计算好的 npz 中读取：
        - fbp_image: 已经过 CNN + FBP + 归一化 的图像
        - ct_label : 对应的真值 CT 图像
    """
    def __init__(self, npz_path, is_train=True, normalize_img=(0, 1)):
        super().__init__()
        self.is_train = is_train
        self.normalize_img = normalize_img

        npz = np.load(npz_path)
        self.fbp = npz["fbp"].astype(np.float32)      # 形状 (N, 1, H, W) 或 (N, H, W)
        self.ct  = npz["ct_label"].astype(np.float32) # 同样形状

        # 统一成 (N, 1, H, W)
        if self.fbp.ndim == 3:
            self.fbp = self.fbp[:, None, ...]
        if self.ct.ndim == 3:
            self.ct = self.ct[:, None, ...]

        assert self.fbp.shape == self.ct.shape, \
            f"fbp 和 ct 的 shape 不一致: {self.fbp.shape} vs {self.ct.shape}"

    def __len__(self):
        return self.fbp.shape[0]

    def __getitem__(self, idx):
        fbp_img = torch.from_numpy(self.fbp[idx])   # (1, H, W)
        ct_img  = torch.from_numpy(self.ct[idx])    # (1, H, W)

        sample = {
            "fbp_image": fbp_img,
            "ct_label": ct_img
        }
        return sample

class FBPCTTestDataset(Dataset):
    """
    测试阶段用的数据集，从 test FBP npz 里读取：
      - fbp_li   : 线性插值 sinogram 的 FBP 结果
      - fbp_cnn  : Projection CNN 输出 sinogram 的 FBP 结果
      - ct_label : 真值 CT
    """
    def __init__(self, npz_path):
        super().__init__()
        npz = np.load(npz_path)
        self.fbp_li  = npz["fbp_li"].astype(np.float32)
        self.fbp_cnn = npz["fbp_cnn"].astype(np.float32)
        self.ct      = npz["ct_label"].astype(np.float32)

        # 统一成 (N, 1, H, W)
        if self.fbp_li.ndim == 3:
            self.fbp_li = self.fbp_li[:, None, ...]
        if self.fbp_cnn.ndim == 3:
            self.fbp_cnn = self.fbp_cnn[:, None, ...]
        if self.ct.ndim == 3:
            self.ct = self.ct[:, None, ...]

        assert self.fbp_li.shape == self.ct.shape, \
            f"fbp_li 和 ct shape 不一致: {self.fbp_li.shape} vs {self.ct.shape}"
        assert self.fbp_cnn.shape == self.ct.shape, \
            f"fbp_cnn 和 ct shape 不一致: {self.fbp_cnn.shape} vs {self.ct.shape}"

    def __len__(self):
        return self.ct.shape[0]

    def __getitem__(self, idx):
        return {
            "fbp_li":   torch.from_numpy(self.fbp_li[idx]),    # (1, H, W)
            "fbp_cnn":  torch.from_numpy(self.fbp_cnn[idx]),   # (1, H, W)
            "ct_label": torch.from_numpy(self.ct[idx])         # (1, H, W)
        }




if __name__ == '__main__':
    print("Testing CTDataset")
    train_path = os.path.join(config.DATA_DIR, config.TRAIN_FILE)
    test_path = os.path.join(config.DATA_DIR, config.TEST_FILE)

    if not os.path.exists(train_path):
         print(f"\nWARNING: Training file not found at {train_path}. Cannot test DataLoader.")
         print("Please ensure 'data/train_meiaonew.npz' exists.")
    else:
        print(f"\nCreating training dataset...")
        train_dataset = CTDataset(npz_path=train_path, sparse_angles_idx=config.SPARSE_VIEW_ANGLES)
        print(f"Training dataset size: {len(train_dataset)}")

        print("\nCreating training DataLoader...")
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0) # Set num_workers > 0 for parallel loading

        print("\nGetting one batch from DataLoader...")
        try:
            batch = next(iter(train_loader))
            sparse_sino = batch['sparse_sinogram']
            full_sino = batch['full_sinogram']
            ct_label = batch['ct_label']

            print(f"  Batch - Sparse Sinogram shape: {sparse_sino.shape}, dtype: {sparse_sino.dtype}, range: [{sparse_sino.min():.2f}, {sparse_sino.max():.2f}]")
            print(f"  Batch - Full Sinogram shape:   {full_sino.shape}, dtype: {full_sino.dtype}, range: [{full_sino.min():.2f}, {full_sino.max():.2f}]")
            print(f"  Batch - CT Label shape:        {ct_label.shape}, dtype: {ct_label.dtype}, range: [{ct_label.min():.2f}, {ct_label.max():.2f}]")

            # Check if sparse sinogram has zeros in non-selected angles
            first_sparse = sparse_sino[0, 0].numpy() # First item in batch, remove channel dim
            non_sparse_angles = np.setdiff1d(np.arange(config.NUM_ANGLES_FULL), config.SPARSE_VIEW_ANGLES)
            print(f"  Sum of values in non-selected angles of sparse sino (should be ~0): {np.sum(np.abs(first_sparse[non_sparse_angles, :])):.4f}")

        except Exception as e:
             print(f"Error getting batch: {e}")
             print("Ensure the .npz file structure and keys ('ct_label', 'sin357') are correct.")

    if not os.path.exists(test_path):
        print(f"\nWARNING: Test file not found at {test_path}.")
    else:
        print(f"\nCreating test dataset...")
        test_dataset = CTDataset(npz_path=test_path, sparse_angles_idx=config.SPARSE_VIEW_ANGLES, is_train=False)
        print(f"Test dataset size: {len(test_dataset)}")