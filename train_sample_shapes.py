import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train nhanh YOLO trên dataset mẫu datasets/sample_shapes (tạo bằng tools/make_sample_dataset.py)."
    )
    parser.add_argument(
        "--data",
        default=str(Path(__file__).resolve().parent / "datasets" / "sample_shapes" / "data.yaml"),
        help="Đường dẫn data.yaml (mặc định: datasets/sample_shapes/data.yaml).",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Base weights để fine-tune (vd: yolo11n.pt / yolov8n.pt).",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Số epochs (mặc định 10).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (mặc định 640).")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (mặc định 8).")
    parser.add_argument("--device", default="cpu", help="cpu hoặc CUDA device id (vd '0').")
    parser.add_argument(
        "--project",
        default=str(Path(__file__).resolve().parent / "runs"),
        help="Thư mục output runs/ (mặc định: ./runs).",
    )
    parser.add_argument("--name", default="sample_shapes", help="Tên run (mặc định sample_shapes).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        print("Không import được ultralytics. Hãy cài requirements.txt trước.")
        print(f"Chi tiết lỗi: {exc}")
        return 2

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        print(f"Không thấy data.yaml: {data_path}")
        print("Hãy chạy: python tools/make_sample_dataset.py")
        return 2

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        verbose=True,
    )

    run_dir = Path(args.project).resolve() / args.name
    weights = run_dir / "weights"
    best_pt = weights / "best.pt"
    last_pt = weights / "last.pt"
    print("Done.")
    print(f"- run_dir: {run_dir}")
    print(f"- best: {best_pt if best_pt.exists() else '(missing)'}")
    print(f"- last: {last_pt if last_pt.exists() else '(missing)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
