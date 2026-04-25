# main.py — grape segmentation research: train YOLOv11 / v12 / v26 + paper figures & tables
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths & hyperparameters (fair comparison ke liye)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "paper_results"
RUNS_DIR = OUTPUT / "training_runs"
FIGURES_DIR = OUTPUT / "figures"
TABLES_DIR = OUTPUT / "tables"
ARTIFACTS_DIR = OUTPUT / "run_snapshots"

EPOCHS = 50
IMG_SIZE = 640
BATCH = 16
# DataLoader workers gpu ko exhaust se bachane ke liye
# the page file (WinError 1455 loading cublas) default value 0 set hai taki late ho pr model pura train ho.
def _default_workers() -> int:
    env = os.environ.get("YOLO_WORKERS")
    if env is not None:
        return max(0, int(env))
    if platform.system() == "Windows":
        return 0
    return min(8, os.cpu_count() or 4)


WORKERS = _default_workers()

DEVICE = 0 if torch.cuda.is_available() else "cpu"

EXPERIMENTS = [
    {
        "key": "yolov11",
        "title": "YOLOv11",
        "weights": "yolo11n-seg.pt",
        "dataset_root": ROOT / "Grape" / "Grape.v1i.yolov11",
    },
    {
        "key": "yolov12",
        "title": "YOLOv12",
        # Ultralytics ke assets me yolo12n-seg.pt available nahi hai (sirf detect checkpoints milte hain), isliye segmentation ke liye YAML config use karna padega.
        "weights": "yolo12n-seg.yaml",
        "dataset_root": ROOT / "Grape" / "Grape.v1i.yolov12",
    },
    {
        "key": "yolov26",
        "title": "YOLOv26",
        "weights": "yolo26n-seg.pt",
        "dataset_root": ROOT / "Grape" / "Grape.v1i.yolo26",
    },
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _data_yaml(dataset_root: Path) -> Path:
    p = dataset_root / "data.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"Missing data.yaml: {p}")
    return p


def _test_images_dir(dataset_root: Path) -> Path:
    d = dataset_root / "test" / "images"
    if not d.is_dir():
        raise FileNotFoundError(f"Missing test images: {d}")
    return d


def _resolve_metric_col(df: pd.DataFrame, stem: str) -> str | None:
    """Pick mask (M) metrics for segmentation; fall back to box (B)."""
    for suffix in ("(M)", "(B)"):
        c = f"metrics/{stem}{suffix}"
        if c in df.columns:
            return c
    return None


def _last_epoch_metrics(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    out = {}
    for stem, label in (
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("mAP50", "mAP50"),
        ("mAP50-95", "mAP50_95"),
    ):
        col = _resolve_metric_col(df, stem)
        out[label] = float(last[col]) if col else float("nan")
    p, r = out["Precision"], out["Recall"]
    out["F1"] = (
        2 * p * r / (p + r)
        if p + r > 0 and not (math.isnan(p) or math.isnan(r))
        else float("nan")
    )
    return out


def _snapshot_run_outputs(run_dir: Path, key: str) -> None:
    dest = ARTIFACTS_DIR / key
    dest.mkdir(parents=True, exist_ok=True)
    if not run_dir.is_dir():
        return
    for f in run_dir.iterdir():
        if f.is_file() and f.suffix.lower() in {".png", ".csv", ".jpg"}:
            shutil.copy2(f, dest / f.name)


def _benchmark_ms_per_image(best_pt: Path, test_images: Path, imgsz: int) -> float:
    paths = sorted(
        p
        for p in test_images.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not paths:
        return float("nan")
    model = YOLO(str(best_pt))
    # Warmup
    for p in paths[: min(5, len(paths))]:
        model.predict(p, imgsz=imgsz, device=DEVICE, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for p in paths:
        model.predict(p, imgsz=imgsz, device=DEVICE, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (elapsed / len(paths)) * 1000.0


def train_one(exp: dict) -> dict:
    key = exp["key"]
    data = _data_yaml(exp["dataset_root"])
    test_imgs = _test_images_dir(exp["dataset_root"])
    run_dir = RUNS_DIR / key
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}\nTraining {exp['title']} ({key})\n  data: {data}\n  weights: {exp['weights']}\n{'=' * 60}\n")

    model = YOLO(exp["weights"])
    model.train(
        data=str(data),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        project=str(RUNS_DIR),
        name=key,
        exist_ok=True,
        device=DEVICE,
        task="segment",
        patience=0,  # patience 0 krhe se model train stop nhi hot hi till the epocs value
    )

    # Ultralytics writes under RUNS_DIR / key
    save_dir = RUNS_DIR / key
    results_csv = save_dir / "results.csv"
    best_pt = save_dir / "weights" / "best.pt"

    _snapshot_run_outputs(save_dir, key)

    inf_ms = float("nan")
    if best_pt.is_file():
        inf_ms = _benchmark_ms_per_image(best_pt, test_imgs, IMG_SIZE)

    row = {
        "key": key,
        "title": exp["title"],
        "results_csv": str(results_csv),
        "run_dir": str(save_dir),
        "best_weights": str(best_pt) if best_pt.is_file() else "",
        "test_images": str(test_imgs),
        "inference_ms_per_image": inf_ms,
    }
    manifest_path = OUTPUT / "manifest.json"
    manifest = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = [m for m in manifest if m.get("key") != key]
    manifest.append(row)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Predictions for qualitative review
    if best_pt.is_file():
        pred_name = f"{key}_test_predictions"
        YOLO(str(best_pt)).predict(
            source=str(test_imgs),
            save=True,
            project=str(OUTPUT / "predictions"),
            name=pred_name,
            device=DEVICE,
            imgsz=IMG_SIZE,
            exist_ok=True,
        )

    return row


def build_summary_table(manifest: list[dict]) -> pd.DataFrame:
    rows = []
    for m in manifest:
        csv_path = Path(m["results_csv"])
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        met = _last_epoch_metrics(df)
        rows.append(
            {
                "Model": m["title"],
                "Key": m["key"],
                "Precision": met["Precision"],
                "Recall": met["Recall"],
                "F1": met["F1"],
                "mAP50": met["mAP50"],
                "mAP50_95": met["mAP50_95"],
                "Inference_ms_per_image": m.get("inference_ms_per_image", float("nan")),
                "Epochs_trained": len(df),
            }
        )
    return pd.DataFrame(rows)


def plot_training_curves(manifest: list[dict]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    series_defs = [
        ("mAP50", "mAP@50"),
        ("mAP50-95", "mAP@50-95"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]

    plt.figure(figsize=(11, 7))
    for stem, label in series_defs:
        for m in manifest:
            csv_path = Path(m["results_csv"])
            if not csv_path.is_file():
                continue
            df = pd.read_csv(csv_path)
            col = _resolve_metric_col(df, stem)
            if not col:
                continue
            plt.plot(df[col].values, label=f"{m['title']} — {label}")
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title("Training curves — grape segmentation (mask metrics)")
    plt.legend(fontsize=8, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p = FIGURES_DIR / "training_curves_all_metrics.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"Saved {p}")


def plot_training_panel(manifest: list[dict]) -> None:
    """2×2 panel (mAP / P / R) — easier to place in a manuscript than one crowded figure."""
    stems = [
        ("mAP50", "mAP@50 (mask)"),
        ("mAP50-95", "mAP@50-95 (mask)"),
        ("precision", "Precision (mask)"),
        ("recall", "Recall (mask)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (stem, title) in zip(axes.ravel(), stems):
        for m in manifest:
            csv_path = Path(m["results_csv"])
            if not csv_path.is_file():
                continue
            df = pd.read_csv(csv_path)
            col = _resolve_metric_col(df, stem)
            if not col:
                continue
            ax.plot(df[col].values, label=m["title"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Grape segmentation — validation metrics vs. epoch")
    plt.tight_layout()
    p = FIGURES_DIR / "training_curves_panel_2x2.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"Saved {p}")


def plot_loss_curves(manifest: list[dict]) -> None:
    """Training losses (when present in results.csv)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    any_plotted = False
    for m in manifest:
        csv_path = Path(m["results_csv"])
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        if "train/box_loss" in df.columns:
            axes[0].plot(df["train/box_loss"].values, label=m["title"])
            any_plotted = True
        if "train/seg_loss" in df.columns:
            axes[1].plot(df["train/seg_loss"].values, label=m["title"])
            any_plotted = True
    axes[0].set_title("Train box loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_title("Train seg loss")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    p = FIGURES_DIR / "training_loss_box_seg.png"
    if any_plotted:
        plt.savefig(p, dpi=200)
        print(f"Saved {p}")
    plt.close()


def plot_final_bars(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["Precision", "Recall", "F1", "mAP50", "mAP50_95"]
    x = range(len(summary))
    width = 0.14
    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        if metric not in summary.columns:
            continue
        plt.bar([xi + i * width for xi in x], summary[metric].values, width, label=metric)
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Final validation metrics (mask) — comparison for publication")
    plt.xticks([v + width * 2 for v in x], summary["Model"].tolist())
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p = FIGURES_DIR / "final_metrics_grouped_bar.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"Saved {p}")

    # Inference time separate (different scale)
    if "Inference_ms_per_image" in summary.columns and summary["Inference_ms_per_image"].notna().any():
        plt.figure(figsize=(8, 5))
        plt.bar(summary["Model"], summary["Inference_ms_per_image"])
        plt.ylabel("ms / image (test set)")
        plt.title("Inference time — GPU batch=1 per image")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        p2 = FIGURES_DIR / "inference_time_bar.png"
        plt.savefig(p2, dpi=200)
        plt.close()
        print(f"Saved {p2}")


def run_pipeline(only_keys: list[str] | None = None, analyze_only: bool = False) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    exps = EXPERIMENTS
    if only_keys:
        sk = set(only_keys)
        exps = [e for e in EXPERIMENTS if e["key"] in sk]
        missing = sk - {e["key"] for e in exps}
        if missing:
            raise SystemExit(f"Unknown --only keys: {missing}. Valid: {[e['key'] for e in EXPERIMENTS]}")

    if torch.cuda.is_available():
        print(
            f"CUDA: {torch.cuda.get_device_name(0)} | "
            f"PyTorch {torch.__version__} | runtime {torch.version.cuda}"
        )
    else:
        print("CUDA unavailable — training on CPU (slow).")

    print(f"DataLoader workers: {WORKERS} (env YOLO_WORKERS=N to change; Windows default 0 avoids WinError 1455)")

    manifest_path = OUTPUT / "manifest.json"
    if analyze_only:
        if not manifest_path.is_file():
            raise SystemExit(f"No {manifest_path}; run training first.")
    else:
        for exp in exps:
            train_one(exp)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    order = {e["key"]: i for i, e in enumerate(EXPERIMENTS)}
    manifest.sort(key=lambda m: order.get(m["key"], 99))

    summary = build_summary_table(manifest)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_out = TABLES_DIR / "summary_metrics.csv"
    summary.to_csv(csv_out, index=False)
    print(f"\nSaved {csv_out}\n{summary.to_string(index=False)}\n")

    plot_training_curves(manifest)
    plot_training_panel(manifest)
    plot_loss_curves(manifest)
    plot_final_bars(summary)

    try:
        from comparison_grid import save_comparison_grid

        _orig_test = ROOT / "Grape" / "Grape.v1i.yolov11" / "test" / "images"
        _grid = save_comparison_grid(_orig_test)
        print(f"Qualitative comparison grid: {_grid}")
    except Exception as e:
        print(f"Qualitative grid skipped: {e}")

    print(
        f"\nOutputs:\n"
        f"  Tables: {TABLES_DIR}\n"
        f"  Figures: {FIGURES_DIR}\n"
        f"  Run PNG/CSV snapshots: {ARTIFACTS_DIR}\n"
        f"  Predictions: {OUTPUT / 'predictions'}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Grape segmentation — YOLOv11/v12/v26 research pipeline")
    parser.add_argument("--only", nargs="*", help=f"Train subset, keys: {[e['key'] for e in EXPERIMENTS]}")
    parser.add_argument("--analyze-only", action="store_true", help="Rebuild tables/figures from manifest + existing results.csv")
    args = parser.parse_args()
    run_pipeline(only_keys=args.only or None, analyze_only=args.analyze_only)


if __name__ == "__main__":
    main()
