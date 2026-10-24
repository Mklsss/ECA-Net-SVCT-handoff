# export_svct_pred_by_indices.py
import os
import argparse
import numpy as np
import torch
from PIL import Image

# ------------- 关键：按你的 main.py 导入项目模块 -------------
import config
from module import Generator  # ProjectionCNN / Discriminator 不需要导出时用


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _squeeze_to_hw(x: np.ndarray) -> np.ndarray:
    """
    支持形状：
      (H,W)
      (1,H,W)
      (H,W,1)
    返回 (H,W)
    """
    x = np.asarray(x)
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        # (1,H,W)
        if x.shape[0] == 1:
            return x[0]
        # (H,W,1)
        if x.shape[-1] == 1:
            return x[..., 0]
    raise ValueError(f"Unsupported image shape: {x.shape}")


def _denorm_to_01(img: np.ndarray) -> np.ndarray:
    """
    将模型/数据集中的图像统一映射到 [0,1] 便于保存 PNG。
    - 若 config.normalize_img == (-1,1)，则 (x+1)/2
    - 否则假定已经在 [0,1]
    """
    img = img.astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

    if getattr(config, "normalize_img", (0, 1)) == (-1, 1):
        img = (img + 1.0) * 0.5

    img = np.clip(img, 0.0, 1.0)
    return img


def _save_png(img01_hw: np.ndarray, path: str):
    img01_hw = _denorm_to_01(img01_hw)
    img_u8 = (img01_hw * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img_u8, mode="L").save(path)


def _extract_state_dict(ckpt_obj):
    """
    兼容常见保存格式：
    1) torch.save(model.state_dict())
    2) torch.save({'state_dict': model.state_dict()})
    3) torch.save({'generator_state_dict': ...})
    4) torch.save({'generator': ...})
    """
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if "generator_state_dict" in ckpt_obj and isinstance(ckpt_obj["generator_state_dict"], dict):
            return ckpt_obj["generator_state_dict"]
        if "generator" in ckpt_obj and isinstance(ckpt_obj["generator"], dict):
            return ckpt_obj["generator"]

        # 很多项目直接就是 state_dict（key=层名，value=tensor）
        # 这里用一个保守判断：value 大多是 Tensor/ndarray/数值
        tensor_like = True
        for k, v in ckpt_obj.items():
            if not isinstance(k, str):
                tensor_like = False
                break
            # v 可能是 torch.Tensor
            if not (torch.is_tensor(v) or isinstance(v, (np.ndarray, float, int))):
                tensor_like = False
                break
        if tensor_like:
            return ckpt_obj

    raise ValueError(
        "无法从 checkpoint 中解析出 state_dict。"
        "建议你在脚本里临时 print(ckpt_obj.keys()) 看一下实际结构。"
    )


def build_generator(device: torch.device) -> torch.nn.Module:
    """
    完全复用你 main.py 里 Generator 的构造方式，避免“猜模型结构”。
    """
    gen = Generator(
        in_channels=config.IMG_CHANNELS_GEN,
        out_channels=config.IMG_CHANNELS_GEN,
        features=config.GEN_FEATURES,
        num_blocks=config.GEN_NUM_BLOCKS,
        bottleneck_expansion=config.GEN_BOTTLENECK_EXPANSION,
        use_eca=config.USE_ECA_IN_GENERATOR,
        eca_k_size=config.ECA_KERNEL_SIZE
    ).to(device)
    gen.eval()
    return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True,
                        help="你的 test_fbp_xxviews.npz（包含 fbp_li / fbp_cnn / ct_label）")
    parser.add_argument("--gen_ckpt", type=str, required=True,
                        help="你的 Generator 权重路径（generator_final_xxviews_xxx.pth 或 generator_epoch_xxx.pth）")
    parser.add_argument("--indices", type=int, nargs="+", required=True,
                        help="要导出的样本 idx 列表，例如 0 66 133 199")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="导出目录，例如 /autodl-fs/data/exports/svct")
    parser.add_argument("--tag", type=str, default="",
                        help="可选：在输出文件名前加一个 tag，便于区分不同模型")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")
    print("[Export] device =", device)

    if not os.path.exists(args.npz):
        raise FileNotFoundError(f"NPZ not found: {args.npz}")
    if not os.path.exists(args.gen_ckpt):
        raise FileNotFoundError(f"Generator ckpt not found: {args.gen_ckpt}")

    data = np.load(args.npz)
    keys = list(data.keys())
    print("[Export] NPZ keys =", keys)

    # 你这套流程里 test npz 正常应该有这三个
    if "ct_label" not in data:
        raise KeyError("NPZ 缺少 ct_label。请确认你传入的是 test_fbp_xxviews.npz。")
    if "fbp_cnn" not in data:
        raise KeyError("NPZ 缺少 fbp_cnn。请确认你传入的是 test_fbp_xxviews.npz。")
    if "fbp_li" not in data:
        raise KeyError("NPZ 缺少 fbp_li。请确认你传入的是 test_fbp_xxviews.npz。")

    ct_all = data["ct_label"]   # 可能是 (N,1,H,W) 或 (N,H,W) 或 (N,H,W,1)
    cnn_all = data["fbp_cnn"]
    li_all = data["fbp_li"]

    # 统一到 (N, H, W)
    def to_nhw(x):
        x = np.asarray(x)
        if x.ndim == 4 and x.shape[1] == 1:      # (N,1,H,W)
            return x[:, 0, :, :].astype(np.float32)
        if x.ndim == 4 and x.shape[-1] == 1:     # (N,H,W,1)
            return x[:, :, :, 0].astype(np.float32)
        if x.ndim == 3:                          # (N,H,W)
            return x.astype(np.float32)
        raise ValueError(f"Unsupported array shape: {x.shape}")

    ct_all = to_nhw(ct_all)
    cnn_all = to_nhw(cnn_all)
    li_all = to_nhw(li_all)

    n = ct_all.shape[0]
    for idx in args.indices:
        if idx < 0 or idx >= n:
            raise IndexError(f"Index out of range: {idx} (dataset size={n})")

    # 构建并加载 Generator（严格按 main.py 参数）
    gen = build_generator(device)
    ckpt_obj = torch.load(args.gen_ckpt, map_location=device)
    state_dict = _extract_state_dict(ckpt_obj)

    # 这里用 strict=True；如果你确实改过结构导致 key 对不上，再改 strict=False
    gen.load_state_dict(state_dict, strict=True)
    gen.eval()

    # 输出目录结构
    # out_dir/
    #   gt/
    #   fbp_li/
    #   fbp_cnn/
    #   svct_pred/
    out_gt = os.path.join(args.out_dir, "gt")
    out_li = os.path.join(args.out_dir, "fbp_li")
    out_cnn = os.path.join(args.out_dir, "fbp_cnn")
    out_pred = os.path.join(args.out_dir, "svct_pred")
    _ensure_dir(out_gt)
    _ensure_dir(out_li)
    _ensure_dir(out_cnn)
    _ensure_dir(out_pred)

    tag = (args.tag + "_") if args.tag else ""

    with torch.no_grad():
        for idx in args.indices:
            gt = ct_all[idx]
            fbp_li = li_all[idx]
            fbp_cnn = cnn_all[idx]

            # Generator 输入：按你训练时的写法，输入就是 fbp_cnn（已经归一化到 config.normalize_img）
            x = torch.from_numpy(fbp_cnn).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
            y = gen(x)
            y_np = y.detach().cpu().numpy()[0, 0, :, :]

            # 保存 PNG（统一映射到 [0,1]）
            _save_png(gt, os.path.join(out_gt,   f"{tag}idx_{idx:04d}.png"))
            _save_png(fbp_li, os.path.join(out_li,  f"{tag}idx_{idx:04d}.png"))
            _save_png(fbp_cnn, os.path.join(out_cnn, f"{tag}idx_{idx:04d}.png"))
            _save_png(y_np, os.path.join(out_pred, f"{tag}idx_{idx:04d}.png"))

            print(f"[OK] idx={idx} exported.")

    print("\nAll done.")
    print("Export dir =", args.out_dir)


if __name__ == "__main__":
    main()
