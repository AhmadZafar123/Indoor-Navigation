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
import sys
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "models" / "trained" / "homeobjects_baseline_v1" / "best.pt"
DEFAULT_OUTPUT = ROOT / "detected-images"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feedbacks.tts import ObjectAnnouncer
from src.decision.fov import FieldOfView


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
    announcer: ObjectAnnouncer | None = None,
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
    print("Camera running. Controls: q/ESC quit, s save snapshot, p pause, t toggle speech.\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    fov = FieldOfView(width_ratio=0.4, height_ratio=0.8)

    frame_index = 0
    paused = False
    last_result = None
    last_annotated = None
    detected_classes: set[str] = set()

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
                result = results[0]
                img_height, img_width = result.orig_shape

                detected_classes = set()
                in_fov_detections: list[dict] = []
                for box in result.boxes:
                    if not fov.is_inside_fov(box, img_width, img_height):
                        continue
                    cls_id = int(box.cls.item())
                    class_name = result.names[cls_id]
                    detected_classes.add(class_name)
                    in_fov_detections.append({
                        "class_name": class_name,
                        "confidence": float(box.conf.item()),
                    })

                last_result = result
                annotated_frame = result.plot()  # BGR numpy array
                last_annotated = fov.draw_fov(annotated_frame)
                frame_index += 1

                if announcer is not None:
                    announcer.update(in_fov_detections)

            if last_annotated is None:
                continue

            display = last_annotated.copy()
            status = "PAUSED" if paused else "LIVE"
            speech = "ON" if announcer is not None and announcer.enabled else "OFF"
            cv2.putText(
                display,
                f"{status}  frame {frame_index}  speech {speech}  "
                f"in-path {len(detected_classes)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                paused = not paused
                print("Paused." if paused else "Resumed.")
            elif key == ord("t") and announcer is not None:
                print("Speech on." if announcer.toggle() else "Speech off.")
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

    parser.add_argument("--no-tts", action="store_true",
                        help="Disable spoken object announcements")
    parser.add_argument("--tts-conf", type=float, default=0.4,
                        help="Minimum confidence to announce an object (default: 0.4)")
    parser.add_argument("--tts-cooldown", type=float, default=60.0,
                        help="Seconds before repeating an object that is still in view "
                             "(default: 60); new objects are always announced instantly")
    parser.add_argument("--tts-rate", type=int, default=180,
                        help="Speech rate in words per minute")
    parser.add_argument("--tts-voice", default=None,
                        help="Voice name substring or index (see src/feedbacks/tts.py --list-voices)")
    parser.add_argument("--tts-phrase", default="{name}",
                        help="Speech template, e.g. '{name} detected'")

    args = parser.parse_args()

    weights = resolve_weights(args.weights)

    voice = args.tts_voice
    if isinstance(voice, str) and voice.isdigit():
        voice = int(voice)

    announcer = None
    if not args.no_tts:
        announcer = ObjectAnnouncer(
            min_confidence=args.tts_conf,
            cooldown=args.tts_cooldown,
            rate=args.tts_rate,
            voice=voice,
            phrase=args.tts_phrase,
        )
        print(f"Speech feedback on (press 't' to mute). "
              f"New objects announced instantly; repeated after "
              f"{args.tts_cooldown:g}s while still in view.")

    try:
        run(
            weights=weights,
            output_dir=args.output_dir,
            camera=args.camera,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            save_json=not args.no_json,
            announcer=announcer,
        )
    finally:
        if announcer is not None:
            announcer.close()


if __name__ == "__main__":
    main()
