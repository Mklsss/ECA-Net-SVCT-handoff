# build_tables.py
# Robust assembler: auto-detect view folders (view_30 / view_030 both OK)
# and allow fallback if a specific expected file is missing.

import os
import re
from PIL import Image, ImageDraw, ImageFont

EXPORT_ROOT = "/autodl-fs/data/exports"
OUT_DIR = os.path.join(EXPORT_ROOT, "_tables")

# 如果你只想固定这四个 idx，就保留；否则脚本也会从 gt/ 自动发现。
IDX_LIST_DEFAULT = ["0000", "0066", "0133", "0199"]

# 列定义： (列标题, 路径模板)
COLUMNS = [
    ("GT",        "{root}/gt/idx_{idx}.png"),
    ("FBP_LI",    "{root}/svct/{view}/fbp_li/idx_{idx}.png"),
    ("FBP_CNN",   "{root}/svct/{view}/fbp_cnn/idx_{idx}.png"),
    ("SVCT",      "{root}/svct/{view}/svct_pred/idx_{idx}.png"),
    ("REDCNN",    "{root}/redcnn/{view}/idx_{idx}.png"),
    ("DuDoTrans", "{root}/dudotrans/{view}/idx_{idx}.png"),
]

# 视觉参数
CELL_SIZE = 512
PAD = 12
TOP_BAR = 80
LEFT_BAR = 140
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)

def _load_font(size=28):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()

def _open_and_fit(path, size):
    # 这里不直接 raise；让上层能打印诊断信息
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return img

def _list_view_folders():
    """
    自动从 svct 目录下发现 view_* 文件夹（兼容 view_30 / view_030 等）。
    """
    svct_dir = os.path.join(EXPORT_ROOT, "svct")
    if not os.path.isdir(svct_dir):
        raise FileNotFoundError(f"svct dir not found: {svct_dir}")

    views = []
    for name in os.listdir(svct_dir):
        if os.path.isdir(os.path.join(svct_dir, name)) and name.startswith("view_"):
            views.append(name)

    if not views:
        raise RuntimeError(f"No view_* folders found under {svct_dir}")

    # 按数字排序：view_30 < view_120
    def view_key(v):
        m = re.search(r"view_(\d+)", v)
        return int(m.group(1)) if m else 10**9

    views = sorted(views, key=view_key)
    return views

def _detect_indices():
    """
    如果 gt 里存在 idx_*.png，则从 gt 自动读取；否则用默认 IDX_LIST_DEFAULT。
    """
    gt_dir = os.path.join(EXPORT_ROOT, "gt")
    if not os.path.isdir(gt_dir):
        return IDX_LIST_DEFAULT

    idxs = []
    for fn in os.listdir(gt_dir):
        m = re.match(r"idx_(\d+)\.png$", fn)
        if m:
            idxs.append(m.group(1))

    if not idxs:
        return IDX_LIST_DEFAULT

    idxs = sorted(idxs, key=lambda x: int(x))
    # 如果默认那四张都在，就按默认顺序；否则取前 4 张（你也可以改成全取）
    if all(x in idxs for x in IDX_LIST_DEFAULT):
        return IDX_LIST_DEFAULT
    return idxs[:4]

def _make_missing_tile(size, label):
    img = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    font = _load_font(24)
    txt = f"MISSING\n{label}"
    draw.multiline_text((20, 20), txt, fill=(180, 0, 0), font=font, spacing=8)
    return img

def build_table_for_view(view_name, idx_list):
    os.makedirs(OUT_DIR, exist_ok=True)

    n_rows = len(idx_list)
    n_cols = len(COLUMNS)

    W = LEFT_BAR + n_cols * CELL_SIZE + (n_cols + 1) * PAD
    H = TOP_BAR + n_rows * CELL_SIZE + (n_rows + 1) * PAD

    canvas = Image.new("RGB", (W, H), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    font_title = _load_font(28)
    font_small = _load_font(24)

    # 顶部列标题
    for c, (title, _) in enumerate(COLUMNS):
        x0 = LEFT_BAR + PAD + c * (CELL_SIZE + PAD)
        tw, th = draw.textbbox((0, 0), title, font=font_title)[2:]
        tx = x0 + (CELL_SIZE - tw) // 2
        ty = (TOP_BAR - th) // 2
        draw.text((tx, ty), title, fill=TEXT_COLOR, font=font_title)

    # 行标签 + 填图
    for r, idx in enumerate(idx_list):
        y0 = TOP_BAR + PAD + r * (CELL_SIZE + PAD)

        row_label = f"idx_{idx}"
        tw, th = draw.textbbox((0, 0), row_label, font=font_small)[2:]
        lx = (LEFT_BAR - tw) // 2
        ly = y0 + (CELL_SIZE - th) // 2
        draw.text((lx, ly), row_label, fill=TEXT_COLOR, font=font_small)

        for c, (_, path_tmpl) in enumerate(COLUMNS):
            x0 = LEFT_BAR + PAD + c * (CELL_SIZE + PAD)
            path = path_tmpl.format(root=EXPORT_ROOT, view=view_name, idx=idx)

            img = _open_and_fit(path, CELL_SIZE)
            if img is None:
                # 关键：如果你的真实目录是 view_30 而 view_name 是 view_030（反之亦然），这里做一次兜底替换
                alt_view = None
                m = re.search(r"view_(\d+)", view_name)
                if m:
                    num = int(m.group(1))
                    # 两种格式互转
                    if view_name == f"view_{num:03d}":
                        alt_view = f"view_{num}"
                    else:
                        alt_view = f"view_{num:03d}"

                if alt_view is not None:
                    alt_path = path_tmpl.format(root=EXPORT_ROOT, view=alt_view, idx=idx)
                    img = _open_and_fit(alt_path, CELL_SIZE)
                    if img is not None:
                        path = alt_path  # 仅用于缺失标签显示

            if img is None:
                img = _make_missing_tile(CELL_SIZE, os.path.relpath(path, EXPORT_ROOT))

            canvas.paste(img, (x0, y0))

    out_path = os.path.join(OUT_DIR, f"table_{view_name}.png")
    canvas.save(out_path)
    print(f"[OK] Saved: {out_path}")

def main():
    views = _list_view_folders()
    idx_list = _detect_indices()

    print("[INFO] Detected views:", views)
    print("[INFO] Using indices:", idx_list)

    for v in views:
        build_table_for_view(v, idx_list)

if __name__ == "__main__":
    main()
