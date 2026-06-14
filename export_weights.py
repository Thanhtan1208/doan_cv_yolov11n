import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export weights YOLO (vd ONNX) từ một file .pt."
    )
    parser.add_argument(
        "--weights",
        default=str(Path(__file__).resolve().parent / "runs" / "sample_shapes_ft" / "weights" / "best.pt"),
        help="Đường dẫn .pt (mặc định best.pt của latest sample_shapes_ft run).",
    )
    parser.add_argument("--format", default="onnx", help="Format export (mặc định onnx).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (mặc định 640).")
    parser.add_argument("--device", default="cpu", help="cpu hoặc CUDA device id (vd '0').")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        print(f"Không thấy weights: {weights_path}")
        print("Gợi ý: chạy train_sample_shapes.py trước.")
        return 2

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        print("Không import được ultralytics. Hãy cài requirements.txt trước.")
        print(f"Chi tiết lỗi: {exc}")
        return 2

    model = YOLO(str(weights_path))
    out = model.export(format=args.format, imgsz=args.imgsz, device=args.device)
    print(f"OK: exported -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
