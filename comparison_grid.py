# Build "Original | Model A | Model B | …" qualitative grid for papers (matplotlib-free tiles via PIL).
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "paper_results"
FIGURES_DIR = OUTPUT / "figures"
PRED_ROOT = OUTPUT / "predictions"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_ROWS: list[tuple[str, Path | None]] = [
    ("Original images", None),
    ("YOLOv11", PRED_ROOT / "yolov11_test_predictions"),
    ("YOLOv12", PRED_ROOT / "yolov12_test_predictions"),
    ("YOLOv26", PRED_ROOT / "yolov26_test_predictions"),
]


def find_image_by_stem(folder: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = folder / f"{stem}{ext}"
        if p.is_file():
            return p
    for p in folder.glob(f"{stem}.*"):
        if p.suffix.lower() in IMAGE_EXTS:
            return p
    return None


def _letterbox(im: Image.Image, size: tuple[int, int], bg: tuple[int, int, int] = (248, 248, 248)) -> Image.Image:
    w, h = size
    im = im.convert("RGB")
    src_w, src_h = im.size
    scale = min(w / src_w, h / src_h)
    nw, nh = max(1, int(src_w * scale)), max(1, int(src_h * scale))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, bg)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    canvas.paste(im, (ox, oy))
    return canvas


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for fp in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        p = Path(fp)
        if p.is_file():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def save_comparison_grid(
    original_dir: Path,
    rows: list[tuple[str, Path | None]] | None = None,
    num_cols: int = 6,
    cell_size: tuple[int, int] = (320, 240),
    label_col_width: int = 200,
    pad: int = 6,
    out_path: Path | None = None,
) -> Path:
    """
    rows: (title, pred_folder_or_None). None = use original_dir for that row.
    Only columns where the stem exists under original_dir and every non-None pred folder are kept.
    """
    rows = rows or DEFAULT_ROWS
    if not original_dir.is_dir():
        raise FileNotFoundError(f"Original images not found: {original_dir}")

    originals = sorted(
        p for p in original_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    pred_folders = [r[1] for r in rows if r[1] is not None]

    stems: list[str] = []
    for p in originals:
        stem = p.stem
        ok = True
        for folder in pred_folders:
            if not folder.is_dir() or find_image_by_stem(folder, stem) is None:
                ok = False
                break
        if ok:
            stems.append(stem)
    stems = sorted(set(stems))[:num_cols]

    if not stems:
        raise FileNotFoundError(
            "No image stems matched across originals and all prediction folders. "
            f"Check {original_dir} and {pred_folders}."
        )

    cell_w, cell_h = cell_size
    font = _font(18)
    font_small = _font(14)

    nrows, ncols = len(rows), len(stems)
    grid_w = label_col_width + pad + ncols * (cell_w + pad)
    grid_h = pad + nrows * (cell_h + pad)

    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for ri, (label, pred_dir) in enumerate(rows):
        y0 = pad + ri * (cell_h + pad)
        # Row label (wrap long text)
        draw.text((8, y0 + cell_h // 2 - 10), label, fill=(20, 20, 20), font=font)
        for ci, stem in enumerate(stems):
            x0 = label_col_width + pad + ci * (cell_w + pad)
            if pred_dir is None:
                src = find_image_by_stem(original_dir, stem)
            else:
                src = find_image_by_stem(pred_dir, stem)
            if src is None:
                tile = Image.new("RGB", (cell_w, cell_h), (220, 220, 220))
                tdraw = ImageDraw.Draw(tile)
                tdraw.text((10, cell_h // 2), "missing", fill=(100, 100, 100), font=font_small)
            else:
                tile = _letterbox(Image.open(src), (cell_w, cell_h))
            canvas.paste(tile, (x0, y0))

    out = out_path or (FIGURES_DIR / "qualitative_comparison_grid.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, "PNG", dpi=(200, 200))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Original + YOLO prediction grid for papers")
    parser.add_argument(
        "--original",
        type=Path,
        default=ROOT / "Grape" / "Grape.v1i.yolov11" / "test" / "images",
        help="Folder with original test images (column alignment by filename stem)",
    )
    parser.add_argument("--cols", type=int, default=6, help="Number of image columns")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=FIGURES_DIR / "qualitative_comparison_grid.png",
    )
    args = parser.parse_args()

    p = save_comparison_grid(args.original, num_cols=args.cols, out_path=args.output)
    print(f"Saved comparison grid: {p}")


if __name__ == "__main__":
    main()
