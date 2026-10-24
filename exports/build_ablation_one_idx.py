# build_ablation_one_idx.py
import os
import re
import argparse
from PIL import Image, ImageOps, ImageDraw, ImageFont

def _load_gray(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing image: {path}")
    return Image.open(path).convert("L")

def _fit(img, target_w, target_h):
    img = ImageOps.contain(img, (target_w, target_h))
    canvas = Image.new("L", (target_w, target_h), color=0)
    x = (target_w - img.size[0]) // 2
    y = (target_h - img.size[1]) // 2
    canvas.paste(img, (x, y))
    return canvas

def _try_font(size=22):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()

def _pick_any_idx(folder):
    # 在 folder 中找 idx_XXXX.png，随便挑一个（按文件名排序后取第一个）
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = sorted([f for f in os.listdir(folder) if re.match(r"idx_\d{4}\.png$", f)])
    if not files:
        raise FileNotFoundError(f"No idx_XXXX.png found in: {folder}")
    m = re.match(r"idx_(\d{4})\.png$", files[0])
    return int(m.group(1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/autodl-fs/data/exports/svct",
                        help="SVCT导出根目录，包含 view_30/view_60/view_90/view_120")
    parser.add_argument("--eca_subdir", type=str, default="svct_pred",
                        help="有ECA版本所在子目录名")
    parser.add_argument("--noeca_subdir", type=str, default="noeca",
                        help="无ECA版本所在子目录名")
    parser.add_argument("--idx", type=int, default=-1,
                        help="指定idx（四位数原值，如 66）。不指定则自动挑一个。")
    parser.add_argument("--out", type=str, default="/autodl-fs/data/exports/ablation_eca_vs_noeca_oneidx.png",
                        help="输出拼图路径")
    parser.add_argument("--tile_size", type=int, default=420)
    parser.add_argument("--pad", type=int, default=18)
    parser.add_argument("--header_h", type=int, default=60)
    parser.add_argument("--rowlabel_w", type=int, default=120)
    args = parser.parse_args()

    views = [30, 60, 90, 120]

    # 自动挑 idx：从 view_30 的有ECA目录里选一个存在的 idx
    if args.idx < 0:
        probe_dir = os.path.join(args.root, "view_30", args.eca_subdir)
        idx = _pick_any_idx(probe_dir)
    else:
        idx = int(args.idx)

    # 读 2x4 图
    T = int(args.tile_size)
    grid = [[None for _ in range(4)] for _ in range(2)]

    for c, v in enumerate(views):
        p_eca = os.path.join(args.root, f"view_{v}", args.eca_subdir, f"idx_{idx:04d}.png")
        p_no  = os.path.join(args.root, f"view_{v}", args.noeca_subdir, f"idx_{idx:04d}.png")
        grid[0][c] = _fit(_load_gray(p_eca), T, T)
        grid[1][c] = _fit(_load_gray(p_no),  T, T)

    # 画布
    W = args.rowlabel_w + 4 * T + 5 * args.pad
    H = args.header_h + 2 * T + 3 * args.pad
    canvas = Image.new("L", (W, H), color=255)
    draw = ImageDraw.Draw(canvas)

    font_col = _try_font(24)
    font_row = _try_font(24)
    font_note = _try_font(18)

    # 列标题
    for c, v in enumerate(views):
        x0 = args.rowlabel_w + args.pad + c * (T + args.pad)
        y0 = args.pad
        title = f"{v} views"
        tw, th = draw.textbbox((0, 0), title, font=font_col)[2:]
        draw.text((x0 + (T - tw)//2, y0 + (args.header_h - th)//2), title, fill=0, font=font_col)

    # 行标题
    row_titles = ["With ECA", "No ECA"]
    for r in range(2):
        y0 = args.header_h + args.pad + r * (T + args.pad)
        title = row_titles[r]
        tw, th = draw.textbbox((0, 0), title, font=font_row)[2:]
        draw.text(((args.rowlabel_w - tw)//2, y0 + (T - th)//2), title, fill=0, font=font_row)

    # 粘贴图
    for r in range(2):
        for c in range(4):
            x = args.rowlabel_w + args.pad + c * (T + args.pad)
            y = args.header_h + args.pad + r * (T + args.pad)
            canvas.paste(grid[r][c], (x, y))

    # 右下角标注选用的idx（不想要可以删掉这三行）
    note = f"idx_{idx:04d}"
    tw, th = draw.textbbox((0, 0), note, font=font_note)[2:]
    draw.text((W - tw - args.pad, H - th - args.pad), note, fill=0, font=font_note)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    canvas.save(args.out)
    print("[OK] idx used:", idx)
    print("[OK] saved to:", args.out)

if __name__ == "__main__":
    main()
