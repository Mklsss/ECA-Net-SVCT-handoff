import os
import argparse
from PIL import Image, ImageDraw, ImageFont

EXPORT_ROOT = "/autodl-fs/data/exports"
OUT_DIR = os.path.join(EXPORT_ROOT, "_tables")

BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
MISSING_BG = (245, 245, 245)
MISSING_FG = (180, 0, 0)

DEFAULT_VIEW_TO_IDX = {
    30: 0,
    60: 66,
    90: 133,
    120: 199,
}

def load_font(size=28):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()

def resolve_view_folder(root_dir: str, view_num: int) -> str:
    """兼容 view_30 / view_030"""
    cand1 = f"view_{view_num:03d}"
    cand2 = f"view_{view_num}"
    if os.path.isdir(os.path.join(root_dir, cand1)):
        return cand1
    if os.path.isdir(os.path.join(root_dir, cand2)):
        return cand2
    return cand1  # 兜底（后续会 missing）

def open_and_fit(path: str, size: int):
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return img

def make_missing_tile(size: int, label: str):
    img = Image.new("RGB", (size, size), MISSING_BG)
    draw = ImageDraw.Draw(img)
    font = load_font(20)
    draw.multiline_text((16, 16), f"MISSING\n{label}", fill=MISSING_FG, font=font, spacing=6)
    return img

def parse_view_idx_map(s: str):
    """
    解析形如: "30=0,60=66,90=133,120=199"
    """
    mapping = {}
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Invalid mapping item: {p} (expected view=idx)")
        v_str, i_str = p.split("=", 1)
        v = int(v_str.strip())
        i = int(i_str.strip())
        mapping[v] = i
    return mapping

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="",
                        help='Optional mapping: "30=0,60=66,90=133,120=199"')
    parser.add_argument("--views", type=int, nargs="+", default=[30, 60, 90, 120],
                        help="Row views order")
    parser.add_argument("--cell", type=int, default=512, help="tile size (square)")
    parser.add_argument("--out", type=str, default=None, help="output png path")
    parser.add_argument("--rowlabel_w", type=int, default=320,
                        help="left label column width (increase if clipped)")
    parser.add_argument("--header_h", type=int, default=90,
                        help="header height")
    args = parser.parse_args()

    view_to_idx = DEFAULT_VIEW_TO_IDX.copy()
    if args.map.strip():
        view_to_idx.update(parse_view_idx_map(args.map.strip()))

    def idx_str_for(view_num: int) -> str:
        if view_num not in view_to_idx:
            raise ValueError(f"Missing idx mapping for view {view_num}. Use --map to provide it.")
        return f"{view_to_idx[view_num]:04d}"

    col_defs = [
        ("Ground Truth", lambda view_folder, idxs: os.path.join(EXPORT_ROOT, "gt", f"idx_{idxs}.png")),
        ("FBP",          lambda view_folder, idxs: os.path.join(EXPORT_ROOT, "svct", view_folder, "fbp_li",    f"idx_{idxs}.png")),
        ("CNN",          lambda view_folder, idxs: os.path.join(EXPORT_ROOT, "svct", view_folder, "fbp_cnn",   f"idx_{idxs}.png")),
        ("GAN",          lambda view_folder, idxs: os.path.join(EXPORT_ROOT, "svct", view_folder, "svct_pred", f"idx_{idxs}.png")),
        ("REDCNN",       lambda view_folder, idxs: os.path.join(EXPORT_ROOT, "redcnn", view_folder,            f"idx_{idxs}.png")),
        ("DuDoTrans",    lambda view_folder, idxs: os.path.join(EXPORT_ROOT, "dudotrans", view_folder,         f"idx_{idxs}.png")),
    ]

    nrows = len(args.views)
    ncols = len(col_defs)

    cell = args.cell
    pad = 18
    header_h = args.header_h
    rowlabel_w = args.rowlabel_w

    font_col = load_font(28)
    font_row = load_font(30)  # 行标签略大一点，便于看清

    W = rowlabel_w + pad + ncols * (cell + pad)
    H = header_h + pad + nrows * (cell + pad)

    canvas = Image.new("RGB", (W, H), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # 列标题
    for c, (col_name, _) in enumerate(col_defs):
        x0 = rowlabel_w + pad + c * (cell + pad)
        bbox = draw.textbbox((0, 0), col_name, font=font_col)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x0 + (cell - tw) // 2, (header_h - th) // 2),
                  col_name, fill=TEXT_COLOR, font=font_col)

    # 每行
    for r, v in enumerate(args.views):
        idxs = idx_str_for(v)

        svct_view_folder = resolve_view_folder(os.path.join(EXPORT_ROOT, "svct"), v)
        redcnn_view_folder = resolve_view_folder(os.path.join(EXPORT_ROOT, "redcnn"), v)
        dudotrans_view_folder = resolve_view_folder(os.path.join(EXPORT_ROOT, "dudotrans"), v)

        # 行标题：只显示视角，不显示 idx
        row_name = f"{v} views"
        y_row = header_h + pad + r * (cell + pad)

        bbox = draw.textbbox((0, 0), row_name, font=font_row)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # 左侧居中（避免被截断）
        draw.text(((rowlabel_w - tw) // 2, y_row + (cell - th) // 2),
                  row_name, fill=TEXT_COLOR, font=font_row)

        for c, (col_name, path_fn) in enumerate(col_defs):
            x0 = rowlabel_w + pad + c * (cell + pad)

            if col_name in ["FBP", "CNN", "GAN"]:
                view_folder = svct_view_folder
            elif col_name == "REDCNN":
                view_folder = redcnn_view_folder
            elif col_name == "DuDoTrans":
                view_folder = dudotrans_view_folder
            else:
                view_folder = ""

            path = path_fn(view_folder, idxs)
            img = open_and_fit(path, cell)
            if img is None:
                rel = os.path.relpath(path, EXPORT_ROOT)
                img = make_missing_tile(cell, rel)
            canvas.paste(img, (x0, y_row))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = args.out or os.path.join(OUT_DIR, "table_views_models_mappedidx.png")
    canvas.save(out_path)
    print("[OK] view->idx mapping:", {v: f"{view_to_idx[v]:04d}" for v in args.views})
    print("[OK] saved:", out_path)

if __name__ == "__main__":
    main()
