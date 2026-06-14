import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tạo dataset mẫu (synthetic) theo format YOLO cho mục đích demo/train nhanh."
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "datasets" / "sample_shapes"),
        help="Thư mục output dataset (mặc định: datasets/sample_shapes).",
    )
    parser.add_argument("--train", type=int, default=60, help="Số ảnh train (mặc định 60).")
    parser.add_argument("--val", type=int, default=20, help="Số ảnh val (mặc định 20).")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh vuông (mặc định 640).")
    parser.add_argument("--seed", type=int, default=42, help="Seed random (mặc định 42).")
    return parser.parse_args()


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def yolo_box_from_xyxy(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[float, float, float, float]:
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    cx = (x1 + x2) / (2 * w)
    cy = (y1 + y2) / (2 * h)
    return (cx, cy, bw, bh)


def draw_sample(imgsz: int, rng: random.Random) -> tuple[np.ndarray, list[tuple[int, tuple[int, int, int, int]]]]:
    img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    # Nền gradient nhẹ để demo nhìn “thật” hơn
    base = rng.randint(10, 40)
    gx = np.linspace(base, base + rng.randint(15, 45), imgsz, dtype=np.uint8)
    gy = np.linspace(base, base + rng.randint(15, 45), imgsz, dtype=np.uint8)
    img[:, :, 0] = gx[None, :]
    img[:, :, 1] = gy[:, None]
    img[:, :, 2] = base

    annotations: list[tuple[int, tuple[int, int, int, int]]] = []

    # 0: rectangle, 1: circle
    shape_count = rng.randint(1, 3)
    for _ in range(shape_count):
        cls = rng.randint(0, 1)
        color = (rng.randint(120, 255), rng.randint(120, 255), rng.randint(120, 255))

        if cls == 0:
            x1 = rng.randint(0, int(imgsz * 0.65))
            y1 = rng.randint(0, int(imgsz * 0.65))
            x2 = rng.randint(x1 + int(imgsz * 0.15), min(imgsz - 1, x1 + int(imgsz * 0.5)))
            y2 = rng.randint(y1 + int(imgsz * 0.15), min(imgsz - 1, y1 + int(imgsz * 0.5)))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)
            annotations.append((0, (x1, y1, x2, y2)))
        else:
            r = rng.randint(int(imgsz * 0.07), int(imgsz * 0.18))
            cx = rng.randint(r, imgsz - r - 1)
            cy = rng.randint(r, imgsz - r - 1)
            cv2.circle(img, (cx, cy), r, color, thickness=-1)
            annotations.append((1, (cx - r, cy - r, cx + r, cy + r)))

    # Nhiễu nhẹ
    noise = rng.randint(0, 10)
    if noise > 0:
        img = cv2.add(img, np.random.randint(0, noise, img.shape, dtype=np.uint8))

    return img, annotations


def write_split(root: Path, split: str, count: int, imgsz: int, rng: random.Random) -> None:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        img, ann = draw_sample(imgsz, rng)
        stem = f"{split}_{i:04d}"
        img_path = images_dir / f"{stem}.jpg"
        label_path = labels_dir / f"{stem}.txt"

        cv2.imwrite(str(img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        lines: list[str] = []
        h, w = img.shape[:2]
        for cls, (x1, y1, x2, y2) in ann:
            cx, cy, bw, bh = yolo_box_from_xyxy(x1, y1, x2, y2, w, h)
            cx = clamp(cx, 0.0, 1.0)
            cy = clamp(cy, 0.0, 1.0)
            bw = clamp(bw, 0.0, 1.0)
            bh = clamp(bh, 0.0, 1.0)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # data.yaml theo chuẩn Ultralytics
    data_yaml = "\n".join(
        [
            f"path: {root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: rectangle",
            "  1: circle",
            "",
        ]
    )
    (root / "data.yaml").write_text(data_yaml, encoding="utf-8")

    write_split(root, "train", args.train, args.imgsz, rng)
    write_split(root, "val", args.val, args.imgsz, rng)

    print(f"OK: created dataset at {root}")
    print(f"- data: {root / 'data.yaml'}")
    print(f"- train images: {root / 'images' / 'train'}")
    print(f"- val images: {root / 'images' / 'val'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

