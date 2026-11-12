import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO/RT-DETR baseline on the 4-class vehicle dataset.")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data", default="configs/vehicles_4class.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--name", default="yolo11n_4class")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the --model checkpoint (point --model at last.pt). Uses args saved in the checkpoint.",
    )
    args = parser.parse_args()

    model = YOLO(args.model)
    if args.resume:
        model.train(resume=True)
        data = getattr(model.trainer.args, "data", args.data)
    else:
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project="runs/baseline",
            name=args.name,
            exist_ok=True,
            amp=True,
            cache=False,
            plots=False,
        )
        data = args.data
    metrics = model.val(data=data, split="val", device=args.device, verbose=False)
    print(f"FINAL mAP50-95={metrics.box.map:.4f} mAP50={metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
