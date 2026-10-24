# export_gt_by_indices.py
import os
import hashlib
import numpy as np
from PIL import Image

# ======= 配置区：只改这里 =======
NPZ_PATH = "/autodl-fs/data/dataset/test_fbp_120views.npz"
EXPORT_IDXS = [0, 66, 133, 199]
OUT_DIR = "./export_check_gt"
# =================================

def md5_of_array(arr: np.ndarray) -> str:
    m = hashlib.md5()
    m.update(arr.tobytes())
    return m.hexdigest()

def to_uint8_img(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx > mn:
        x = (x - mn) / (mx - mn)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    return x

def load_ct_label(npz_path: str) -> np.ndarray:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    d = np.load(npz_path)
    if "ct_label" not in d.files:
        raise KeyError(f"'ct_label' not found. keys={d.files}")

    ct = d["ct_label"]

    # 统一成 (N, H, W)
    if ct.ndim == 4:
        # (N, 1, H, W) or (N, H, W, 1)
        if ct.shape[1] == 1:
            ct = ct[:, 0, :, :]
        elif ct.shape[-1] == 1:
            ct = ct[:, :, :, 0]
        else:
            raise ValueError(f"Unsupported ct_label shape: {ct.shape}")
    elif ct.ndim != 3:
        raise ValueError(f"Unsupported ct_label shape: {ct.shape}")

    return ct.astype(np.float32)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ct_all = load_ct_label(NPZ_PATH)
    n = ct_all.shape[0]
    print(f"[OK] Loaded ct_label from: {NPZ_PATH}")
    print(f"ct_label shape: {ct_all.shape} (N={n})")
    print(f"Export indices: {EXPORT_IDXS}")
    print("")

    for idx in EXPORT_IDXS:
        if idx < 0 or idx >= n:
            print(f"[SKIP] idx={idx} out of range 0..{n-1}")
            continue

        img = ct_all[idx]  # (H, W)
        h, w = img.shape
        md5 = md5_of_array(img)

        # 保存 png
        img_u8 = to_uint8_img(img)
        save_path = os.path.join(OUT_DIR, f"gt_idx{idx}.png")
        Image.fromarray(img_u8).save(save_path)

        print(f"[SAVE] idx={idx} shape=({h},{w}) min={img.min():.6f} max={img.max():.6f} md5={md5}")
        print(f"       -> {save_path}")

if __name__ == "__main__":
    main()
