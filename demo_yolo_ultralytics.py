import argparse
import time
from pathlib import Path

import cv2


def default_weights() -> str:
    custom = Path(__file__).resolve().parent / "runs" / "sample_shapes" / "weights" / "best.pt"
    return str(custom) if custom.exists() else "yolo11n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo Object Detection bằng Ultralytics YOLO (webcam/video/ảnh)."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Nguồn input: '0' (webcam), hoặc đường dẫn video/ảnh, hoặc URL stream (nếu OpenCV hỗ trợ).",
    )
    parser.add_argument(
        "--model",
        default=default_weights(),
        help="Weights model (ví dụ: yolo11n.pt / yolov8n.pt / runs/sample_shapes/.../best.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kích thước ảnh đầu vào (mặc định 640).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Ngưỡng confidence (mặc định 0.25).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Thiết bị: 'cpu' hoặc '0'/'0,1' cho GPU CUDA (tùy máy).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Lưu video/ảnh kết quả vào thư mục outputs/.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Hiển thị cửa sổ xem kết quả (OpenCV imshow).",
    )
    return parser.parse_args()


def is_int_string(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def ensure_outdir() -> Path:
    outdir = Path(__file__).parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def main() -> int:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        print("Không import được ultralytics. Hãy cài requirements.txt trước.")
        print(f"Chi tiết lỗi: {exc}")
        return 2

    source: int | str
    source = int(args.source) if is_int_string(args.source) else args.source

    model = YOLO(args.model)

    outdir = ensure_outdir() if args.save else None

    # Mẹo trình bày seminar:
    # - Stream inference: model.predict(stream=True) để lấy kết quả từng frame
    # - Dùng `results[0].plot()` để vẽ bbox + nhãn lên ảnh
    start_time = time.time()
    frames = 0

    # Ultralytics cho phép source là webcam index / path / url, tự xử lý đọc.
    # Tuy nhiên để chủ động FPS + hiển thị, ta dùng stream=True và vẽ frame.
    predictions = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        stream=True,
        verbose=False,
    )

    writer = None
    save_path = None

    for result in predictions:
        frames += 1

        annotated = result.plot()

        if args.show:
            cv2.imshow("YOLO Demo", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.save:
            outdir = outdir or ensure_outdir()

            # Nếu input là ảnh -> lưu ảnh; nếu là video/webcam -> lưu mp4
            if result.path and Path(result.path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                save_path = outdir / (Path(result.path).stem + "_pred.jpg")
                cv2.imwrite(str(save_path), annotated)
                print(f"Đã lưu: {save_path}")
                break
            else:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fps = 30
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    save_path = outdir / "demo_pred.mp4"
                    writer = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))
                writer.write(annotated)

    if writer is not None:
        writer.release()
        print(f"Đã lưu: {save_path}")

    if args.show:
        cv2.destroyAllWindows()

    elapsed = max(1e-6, time.time() - start_time)
    fps = frames / elapsed
    print(f"Frames: {frames} | Time: {elapsed:.2f}s | FPS (end-to-end): {fps:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
