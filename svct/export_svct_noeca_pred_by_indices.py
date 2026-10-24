import os
import argparse
import numpy as np
import torch
from PIL import Image

# 你的 SVCT 项目目录里通常有 module.py / config.py；确保运行位置能 import 到
from module import Generator


def _infer_to_01(x: np.ndarray) -> np.ndarray:
    """
    把网络输出映射到 [0,1] 便于存 png。
    兼容两类常见范围：
      - [-1,1] -> (x+1)/2
      - [0,1]  -> 原样
    """
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    mn, mx = float(x.min()), float(x.max())

    # 粗略判断是否是 [-1,1] 体系
    if mn < -0.05:
        x = (x + 1.0) * 0.5

    x = np.clip(x, 0.0, 1.0)
    return x


def _save_png(gray01: np.ndarray, out_path: str):
    """
    gray01: (H,W) in [0,1]
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    u8 = (gray01 * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True, help="test_fbp_xxxviews.npz (包含 fbp_cnn)")
    parser.add_argument("--gen_ckpt", type=str, required=True, help="noeca 生成器权重 .pth")
    parser.add_argument("--views", type=int, required=True, help="30/60/90/120，用于输出目录命名")
    parser.add_argument("--indices", type=int, nargs="+", required=True, help="要导出的 idx 列表")
    parser.add_argument("--out_root", type=str, default="/autodl-fs/data/exports/svct",
                        help="exports/svct 根目录")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    # 下面这些必须与你训练时 Generator 初始化保持一致；不确定就用你 main.py 里的同名默认值
    parser.add_argument("--gen_features", type=int, default=64)
    parser.add_argument("--gen_num_blocks", type=int, default=4)
    parser.add_argument("--gen_bottleneck_expansion", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print("[Device]", device)

    if not os.path.exists(args.npz):
        raise FileNotFoundError(f"NPZ not found: {args.npz}")
    if not os.path.exists(args.gen_ckpt):
        raise FileNotFoundError(f"Generator ckpt not found: {args.gen_ckpt}")

    data = np.load(args.npz)
    if "fbp_cnn" not in data.files:
        raise KeyError(f"npz 缺少 fbp_cnn，当前 keys: {data.files}")

    fbp_cnn = data["fbp_cnn"]  # (N,1,H,W) 或 (N,H,W)
    if fbp_cnn.ndim == 3:
        fbp_cnn = fbp_cnn[:, None, :, :]
    if fbp_cnn.ndim != 4 or fbp_cnn.shape[1] != 1:
        raise ValueError(f"fbp_cnn shape 异常: {fbp_cnn.shape}，期望 (N,1,H,W)")

    n = fbp_cnn.shape[0]
    indices = [i for i in args.indices if 0 <= i < n]
    if not indices:
        raise ValueError("indices 全部越界或为空")

    # 明确：noeca 版本 => use_eca=False
    gen = Generator(
        in_channels=1,
        out_channels=1,
        features=args.gen_features,
        num_blocks=args.gen_num_blocks,
        bottleneck_expansion=args.gen_bottleneck_expansion,
        use_eca=False,
        eca_k_size=3
    ).to(device)

    state = torch.load(args.gen_ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    gen.load_state_dict(state, strict=True)
    gen.eval()
    print("[OK] generator loaded (noeca)")

    view_folder = f"view_{int(args.views)}"
    out_dir = os.path.join(args.out_root, view_folder, "noeca")
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for idx in indices:
            x = torch.from_numpy(fbp_cnn[idx:idx+1]).float().to(device)  # (1,1,H,W)
            y = gen(x)  # (1,1,H,W)
            y_np = y.detach().cpu().numpy()[0, 0]
            y_01 = _infer_to_01(y_np)

            out_path = os.path.join(out_dir, f"idx_{idx:04d}.png")
            _save_png(y_01, out_path)
            print("[Saved]", out_path)

    print("[Done] out_dir =", out_dir)


if __name__ == "__main__":
    main()
