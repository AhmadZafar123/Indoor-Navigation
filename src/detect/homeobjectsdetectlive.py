"""Run YOLOv8 live detection from a webcam.

Usage:
    # Default webcam
    python src/detect/homeobjectsdetectlive.py

    # Second camera and lower confidence
    python src/detect/homeobjectsdetectlive.py --camera 1 --conf 0.3

    # Custom weights
    python src/detect/homeobjectsdetectlive.py --weights models/trained/homeobjects_baseline_v1/best.pt

Controls:
    q / ESC   quit
    s         save the current annotated frame + detection JSON
    p         pause / resume the live feed

Output:
    detected-images/live_<timestamp>.jpg                annotated snapshot
    detected-images/live_<timestamp>_detections.json    detection summary
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "models" / "trained" / "homeobjects_baseline_v1" / "best.pt"
DEFAULT_OUTPUT = ROOT / "detected-images"


def resolve_weights(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {path}.\n"
            "Train a model first or pass --weights path/to/best.pt"
        )
    return path


def detections_to_json(result, frame_index: int) -> dict:
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
        "frame_index": frame_index,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image_size": {
            "width": int(result.orig_shape[1]),
            "height": int(result.orig_shape[0]),
        },
        "num_detections": len(detections),
        "detections": detections,
    }


def save_snapshot(
    annotated, summary: dict, output_dir: Path, save_json: bool
) -> tuple[Path, Path | None]:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
    image_path = output_dir / f"live_{stamp}.jpg"

    cv2.imwrite(str(image_path), annotated)

    json_path = None
    if save_json:
        json_path = output_dir / f"live_{stamp}_detections.json"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return image_path, json_path


def run(
    weights: Path,
    output_dir: Path,
    camera: int,
    conf: float,
    iou: float,
    imgsz: int,
    save_json: bool,
) -> None:
    print(f"Loading model: {weights}")
    model = YOLO(str(weights))

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera}. "
            "Check that it is connected and not in use by another app."
        )

    window_name = "Home Objects - Live Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("Camera running. Controls: q/ESC quit, s save snapshot, p pause.\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    paused = False
    last_result = None
    last_annotated = None

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print("⚠ Camera frame could not be read. Stopping.")
                    break

                results = model.predict(
                    source=frame,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    verbose=False,
                )
                last_result = results[0]
                last_annotated = last_result.plot()  # BGR numpy array
                frame_index += 1

            if last_annotated is None:
                continue

            display = last_annotated.copy()
            status = "PAUSED" if paused else "LIVE"
            cv2.putText(
                display, f"{status}  frame {frame_index}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                paused = not paused
                print("Paused." if paused else "Resumed.")
            elif key == ord("s") and last_result is not None:
                summary = detections_to_json(last_result, frame_index)
                image_path, json_path = save_snapshot(
                    last_annotated, summary, output_dir, save_json
                )
                print(f"  → {image_path.name}  ({summary['num_detections']} detections)")
                if json_path is not None:
                    print(f"  → {json_path.name}")
            elif cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n✅ Done. {frame_index} frame(s) processed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 live webcam detection")
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam index (default: 0)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="Path to trained .pt file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Where to save snapshots (default: detected-images/)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (0-1)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--no-json", action="store_true",
                        help="Skip writing the JSON summary when saving")

    args = parser.parse_args()

    weights = resolve_weights(args.weights)
    run(
        weights=weights,
        output_dir=args.output_dir,
        camera=args.camera,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save_json=not args.no_json,
    )


if __name__ == "__main__":
    main()
