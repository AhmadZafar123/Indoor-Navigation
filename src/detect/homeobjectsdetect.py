"""Run YOLOv8 detection on an image and save annotated output.

Usage:
    # Single image
    python src/perception/run_inference.py --image data/samples/hallway.jpg

    # Whole folder
    python src/perception/run_inference.py --image-dir data/samples/

    # Custom output dir and lower confidence
    python src/perception/run_inference.py --image data/samples/hallway.jpg \
        --output-dir outputs/my_test --conf 0.3

Output:
    outputs/<timestamp>/<image_name>_annotated.jpg     annotated image
    outputs/<timestamp>/<image_name>_detections.json   detection summary
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "models" / "trained" / "homeobjects_baseline_v1" / "best.pt"
DEFAULT_OUTPUT = ROOT / "detected-images"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_weights(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {path}.\n"
            "Train a model first or pass --weights path/to/best.pt"
        )
    return path


def collect_images(image: Path | None, image_dir: Path | None) -> list[Path]:
    if image is not None:
        if not image.exists():
            raise FileNotFoundError(f"Input image not found: {image}")
        return [image]

    if image_dir is not None:
        if not image_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {image_dir}")
        images = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not images:
            raise FileNotFoundError(f"No images found in {image_dir}")
        return images

    raise ValueError("Provide either --image or --image-dir")


def detections_to_json(result) -> dict:
    """Convert an Ultralytics result into a plain dict."""
    names = result.names  # {class_id: class_name}
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        detections.append({
            "class_id": cls_id,
            "class_name": names[cls_id],
            "confidence": round(float(box.conf.item()), 4),
            "bbox_xyxy": [round(float(v), 2) for v in box.xyxy[0].tolist()],
            "bbox_xywh": [round(float(v), 2) for v in box.xywh[0].tolist()],
        })
    return {
        "image": Path(result.path).name,
        "image_size": {
            "width": int(result.orig_shape[1]),
            "height": int(result.orig_shape[0]),
        },
        "num_detections": len(detections),
        "detections": detections,
    }


def run(
    weights: Path,
    images: list[Path],
    output_dir: Path,
    conf: float,
    iou: float,
    imgsz: int,
    save_json: bool,
) -> None:
    print(f"Loading model: {weights}")
    model = YOLO(str(weights))

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {output_dir}\n")

    for img_path in images:
        print(f"Processing: {img_path.name}")
        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
        )

        for result in results:
            annotated = result.plot()  # numpy BGR array with boxes drawn

            # Save annotated image
            stem = Path(result.path).stem
            annotated_path = output_dir / f"{stem}_annotated.jpg"
            # ultralytics has a built-in save; use cv2 for full control
            import cv2
            cv2.imwrite(str(annotated_path), annotated)
            print(f"  → {annotated_path.name}")

            # Save JSON summary
            if save_json:
                summary = detections_to_json(result)
                json_path = output_dir / f"{stem}_detections.json"
                json_path.write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )
                print(f"  → {json_path.name}  ({summary['num_detections']} detections)")

    print(f"\n✅ Done. {len(images)} image(s) processed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 inference on images")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="Path to a single image")
    src.add_argument("--image-dir", type=Path, help="Path to a folder of images")

    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="Path to trained .pt file")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to save results (default: outputs/<timestamp>/)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (0-1)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-json", action="store_true",
                        help="Skip writing the JSON summary")

    args = parser.parse_args()

    weights = resolve_weights(args.weights)
    images = collect_images(args.image, args.image_dir)

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = DEFAULT_OUTPUT / stamp
    else:
        output_dir = args.output_dir

    run(
        weights=weights,
        images=images,
        output_dir=output_dir,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save_json=not args.no_json,
    )


if __name__ == "__main__":
    main()