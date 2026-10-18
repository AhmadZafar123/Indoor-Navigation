"""Train a YOLOv8n detector on the HomeObjects-3K dataset.

Layout:
    data/raw/images/{train,val}       downloaded + extracted dataset
    data/raw/labels/{train,val}
    configs/data/homeobjects.yaml       dataset config pointing at data/raw
    models/pretrained/yolov8n.pt        pretrained weights
    runs/train/homeobjects_yolov8n/     ultralytics run output
    models/trained/homeobjects_baseline_v1/best.pt   copied best checkpoint

Resume:
    Training resumes automatically from the latest interrupted run
    (a weights/last.pt that still has its epoch and optimizer state);
    when several runs were interrupted, the one with the most epochs
    wins. Ultralytics strips that state once a run finishes, so a
    completed run cannot be resumed; training then continues from the
    run's best.pt as a new run instead. Use --fresh to start a new
    run from the pretrained weights.
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO

DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/homeobjects-3K.zip"

CLASS_NAMES = {
    0: "bed",
    1: "sofa",
    2: "chair",
    3: "table",
    4: "lamp",
    5: "tv",
    6: "laptop",
    7: "wardrobe",
    8: "window",
    9: "door",
    10: "potted plant",
    11: "photo frame",
}

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"/ "homeobjects"
CONFIG_PATH = ROOT / "configs" / "data" / "homeobjects.yaml"
PRETRAINED_WEIGHTS = ROOT / "models" / "pretrained" / "yolov8n.pt"
RUN_DIR = ROOT / "runs" / "train"
RUN_NAME = "homeobjects_yolov8n"
BEST_CHECKPOINT = ROOT / "models" / "trained" / "homeobjects_baseline_v1" / "best.pt"


def iter_checkpoints(name: str = "last.pt") -> list[Path]:
    """Return every checkpoint saved by a homeobjects run, oldest first."""
    return sorted(
        RUN_DIR.glob(f"{RUN_NAME}*/weights/{name}"),
        key=lambda p: p.stat().st_ctime,
    )


def find_resumable_checkpoint() -> tuple[Path, dict] | None:
    """Resumable checkpoint with the most epochs, ties broken by newest.

    Ultralytics strips the optimizer and resets epoch to -1 once a run
    finishes, so only checkpoints from interrupted runs can be resumed.
    """
    best_path = None
    best_ckpt = None
    for path in iter_checkpoints():  # oldest first, so ties prefer the newest run
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("epoch", -1) >= 0 and ckpt.get("optimizer") is not None:
            if best_ckpt is None or ckpt.get("epoch", -1) >= best_ckpt.get("epoch", -1):
                best_path, best_ckpt = path, ckpt
    return None if best_path is None else (best_path, best_ckpt)


def find_best_checkpoint() -> Path | None:
    """Newest best.pt across all runs."""
    checkpoints = iter_checkpoints("best.pt")
    return checkpoints[-1] if checkpoints else None


def download_dataset() -> Path:
    if (DATA_RAW / "images" / "train").exists() and (DATA_RAW / "images" / "val").exists():
        print(f"Dataset already present at {DATA_RAW}")
        return DATA_RAW

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_RAW / "homeobjects-3K.zip"

    if not zip_path.exists():
        print(f"Downloading HomeObjects-3K (~390 MB) from {DATASET_URL}")
        with urllib.request.urlopen(DATASET_URL) as resp, open(zip_path, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            with tqdm(total=total, unit="B", unit_scale=True) as pbar:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"Extracting {zip_path} -> {DATA_RAW}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_RAW)
    zip_path.unlink()
    return DATA_RAW


def write_config(dataset_dir: Path) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
    }
    CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Wrote dataset config to {CONFIG_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8n on HomeObjects-3K")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="'0' for GPU, 'cpu', or None to auto-detect")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="start a new run from the pretrained weights instead of resuming",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the latest interrupted run (this is the default behavior)",
    )
    args = parser.parse_args()

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device == "0" else "CPU"
    print(f"Using device: {device} ({gpu_name})")

    dataset_dir = download_dataset()
    write_config(dataset_dir)

    start_weights = None
    resume = False
    if not args.fresh:
        resumable = find_resumable_checkpoint()
        if resumable:
            ckpt_path, ckpt = resumable
            print(f"Found resumable checkpoint: {ckpt_path}")
            total_epochs = (ckpt.get("train_args") or {}).get("epochs", args.epochs)
            print(f"Resuming {ckpt_path.parent.parent.name} from epoch {ckpt['epoch'] + 1} of {total_epochs}")
            resume = True
            start_weights = ckpt_path
        else:
            best = find_best_checkpoint()
            if best is None:
                print("No previous training run found, starting from scratch")
            else:
                print(
                    f"Latest run already completed, so its last.pt was finalized and "
                    f"cannot be resumed. Continuing from {best} as a new run instead."
                )
                start_weights = best

    if start_weights is None:
        PRETRAINED_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        start_weights = PRETRAINED_WEIGHTS  # downloads yolov8n.pt here if missing
    model = YOLO(str(start_weights))

    results = model.train(
        data=str(CONFIG_PATH),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(ROOT / "runs" / "train"),
        name="homeobjects_yolov8n",
        resume=resume,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    BEST_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, BEST_CHECKPOINT)
    print(f"Copied best checkpoint to {BEST_CHECKPOINT}")


if __name__ == "__main__":
    main()
