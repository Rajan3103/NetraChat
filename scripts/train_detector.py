"""
(Optional) Teach the object detector NEW kinds of objects.

The default YOLOv8 model already knows 80 everyday objects. If you need
others (for example "drone", "rifle", "helmet", "PCB component"), you can
fine-tune it on your own labelled images.

1. Collect images and draw boxes around the objects with a free labelling
   tool such as Label Studio, CVAT or Roboflow. Export in "YOLO" format.
2. Arrange the files like this:
       my_dataset/
         images/train/*.jpg     labels/train/*.txt
         images/val/*.jpg       labels/val/*.txt
3. Write my_dataset/data.yaml:
       path: my_dataset
       train: images/train
       val: images/val
       names: {0: drone, 1: helmet}
4. Train:
       python scripts/train_detector.py --data my_dataset/data.yaml --epochs 50
   (A GPU is strongly recommended. On a CPU, use a few epochs to test.)
5. Use the result in the chatbot:
       python api.py --weights runs/detect/netrachat/weights/best.pt

To try the whole process quickly on a tiny sample dataset (downloads ~7 MB):
       python scripts/train_detector.py --data coco8.yaml --epochs 3

New class names also need to be added to netrachat/vocabulary.py
(COCO_FORMS) so the chatbot uses the right singular/plural forms.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from netrachat.vision.detector import MODELS_DIR  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for NetraChat")
    parser.add_argument("--data", required=True, help="dataset YAML file")
    parser.add_argument("--base", default="yolov8s.pt", help="starting weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    from ultralytics import YOLO

    base = Path(args.base)
    if not base.exists() and (MODELS_DIR / args.base).exists():
        base = MODELS_DIR / args.base
    model = YOLO(str(base))
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, project="runs/detect", name="netrachat",
                exist_ok=True)
    metrics = model.val()
    print(f"\nValidation mAP50-95: {metrics.box.map:.3f}   mAP50: {metrics.box.map50:.3f}")
    print("Best weights saved in runs/detect/netrachat/weights/best.pt")


if __name__ == "__main__":
    main()
