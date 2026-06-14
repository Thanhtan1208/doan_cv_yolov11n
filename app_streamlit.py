import csv
import hashlib
import json
import re
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DEFAULT_CUSTOM = APP_DIR / "runs" / "sample_shapes_ft" / "weights" / "best.pt"
if not DEFAULT_CUSTOM.exists():
    DEFAULT_CUSTOM = APP_DIR / "runs" / "sample_shapes" / "weights" / "best.pt"
OUTPUT_DIR = APP_DIR / "outputs"
DATA_DIR = APP_DIR / "data"
HISTORY_FILE = DATA_DIR / "detect_history.jsonl"


@st.cache_resource
def load_model(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(upload) -> Path:
    suffix = Path(upload.name).suffix or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.read())
    tmp.flush()
    return Path(tmp.name)


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def safe_stem(name: str | None, fallback: str) -> str:
    if not name:
        return fallback
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or fallback


def fingerprint_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def image_bytes_from_bgr(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Không mã hóa được ảnh kết quả.")
    return buf.tobytes()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    rows: list[dict] = []
    for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_history(entry: dict) -> None:
    ensure_dirs()
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def extract_detections(result, frame_index: int | None = None) -> list[dict]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    names = getattr(result, "names", {}) or {}
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    detections: list[dict] = []
    for idx in range(len(classes)):
        cls_id = int(classes[idx])
        detections.append(
            {
                "frame": frame_index,
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "confidence": round(float(confs[idx]), 4),
                "bbox": [round(float(v), 2) for v in xyxy[idx].tolist()],
            }
        )
    return detections


def build_summary(
    *,
    kind: str,
    source_name: str,
    weights: str,
    imgsz: int,
    conf: float,
    device: str,
    detections: list[dict],
    output_file: Path,
    extra: dict | None = None,
) -> dict:
    class_counts = Counter(det["class_name"] for det in detections)
    confidences = [float(det["confidence"]) for det in detections]
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "kind": kind,
        "source": source_name,
        "weights": weights,
        "imgsz": imgsz,
        "conf_threshold": conf,
        "device": device,
        "output_file": output_file.name,
        "total_detections": len(detections),
        "unique_classes": len(class_counts),
        "class_counts": dict(class_counts),
        "avg_confidence": round(mean(confidences), 4) if confidences else None,
        "max_confidence": round(max(confidences), 4) if confidences else None,
        "notes": "No objects detected." if not detections else "Detections stored for download and history.",
    }
    if extra:
        summary.update(extra)
    return summary


def process_image(
    model,
    image_bgr: np.ndarray,
    *,
    source_name: str,
    weights: str,
    imgsz: int,
    conf: float,
    device: str,
) -> tuple[np.ndarray, Path, Path, dict]:
    results = model.predict(image_bgr, imgsz=imgsz, conf=conf, device=device, verbose=False)
    result = results[0]
    annotated = result.plot()

    tag = f"{timestamp_tag()}_{safe_stem(source_name, 'image')}"
    out_path = OUTPUT_DIR / f"{tag}_pred.png"
    report_path = OUTPUT_DIR / f"{tag}_report.json"

    cv2.imwrite(str(out_path), annotated)
    detections = extract_detections(result)
    summary = build_summary(
        kind="image",
        source_name=source_name,
        weights=weights,
        imgsz=imgsz,
        conf=conf,
        device=device,
        detections=detections,
        output_file=out_path,
    )
    write_json(report_path, {"summary": summary, "detections": detections})
    append_history(
        {
            "time": summary["timestamp"],
            "kind": "image",
            "source": source_name,
            "output_file": str(out_path.relative_to(APP_DIR)),
            "report_file": str(report_path.relative_to(APP_DIR)),
            "total_detections": summary["total_detections"],
            "unique_classes": summary["unique_classes"],
            "class_counts": summary["class_counts"],
            "avg_confidence": summary["avg_confidence"],
            "max_confidence": summary["max_confidence"],
            "notes": summary["notes"],
        }
    )
    return annotated, out_path, report_path, summary


def process_video(
    model,
    video_path: Path,
    *,
    source_name: str,
    weights: str,
    imgsz: int,
    conf: float,
    device: str,
) -> tuple[Path, Path, Path, dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Không mở được video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    tag = f"{timestamp_tag()}_{safe_stem(source_name, 'video')}"
    out_path = OUTPUT_DIR / f"{tag}_pred.mp4"
    csv_path = OUTPUT_DIR / f"{tag}_detections.csv"
    report_path = OUTPUT_DIR / f"{tag}_report.json"

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    all_detections: list[dict] = []
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            results = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)
            result = results[0]
            annotated = result.plot()
            writer.write(annotated)
            all_detections.extend(extract_detections(result, frame_index=frame_index))
    finally:
        cap.release()
        writer.release()

    summary = build_summary(
        kind="video",
        source_name=source_name,
        weights=weights,
        imgsz=imgsz,
        conf=conf,
        device=device,
        detections=all_detections,
        output_file=out_path,
        extra={
            "source_fps": round(float(fps), 4),
            "frames_processed": frame_index,
            "estimated_duration_s": round(frame_index / float(fps), 4) if fps else None,
        },
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(
            f,
            fieldnames=["frame", "class_id", "class_name", "confidence", "bbox"],
        )
        writer_csv.writeheader()
        for row in all_detections:
            writer_csv.writerow(row)

    write_json(report_path, {"summary": summary, "detections": all_detections})
    append_history(
        {
            "time": summary["timestamp"],
            "kind": "video",
            "source": source_name,
            "output_file": str(out_path.relative_to(APP_DIR)),
            "report_file": str(report_path.relative_to(APP_DIR)),
            "csv_file": str(csv_path.relative_to(APP_DIR)),
            "total_detections": summary["total_detections"],
            "unique_classes": summary["unique_classes"],
            "class_counts": summary["class_counts"],
            "avg_confidence": summary["avg_confidence"],
            "max_confidence": summary["max_confidence"],
            "frames_processed": frame_index,
            "source_fps": summary["source_fps"],
            "notes": summary["notes"],
        }
    )
    return out_path, csv_path, report_path, summary


def show_image_result(prefix: str, annotated: np.ndarray, out_path: Path, report_path: Path, summary: dict) -> None:
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Prediction", use_container_width=True)
    st.download_button(
        "Tải ảnh kết quả",
        data=image_bytes_from_bgr(annotated),
        file_name=out_path.name,
        mime="image/png",
        use_container_width=True,
        key=f"{prefix}_image_download",
    )
    st.download_button(
        "Tải report JSON",
        data=report_path.read_bytes(),
        file_name=report_path.name,
        mime="application/json",
        use_container_width=True,
        key=f"{prefix}_image_report_download",
    )
    st.caption(
        f"Objects: {summary['total_detections']} | Classes: {summary['unique_classes']} | Avg conf: {summary['avg_confidence']}"
    )


def show_video_result(out_path: Path, csv_path: Path, report_path: Path, summary: dict) -> None:
    st.video(str(out_path))
    st.download_button(
        "Tải video kết quả",
        data=out_path.read_bytes(),
        file_name=out_path.name,
        mime="video/mp4",
        use_container_width=True,
        key="video_download",
    )
    st.download_button(
        "Tải CSV danh sách detect",
        data=csv_path.read_bytes(),
        file_name=csv_path.name,
        mime="text/csv",
        use_container_width=True,
        key="video_csv_download",
    )
    st.download_button(
        "Tải report JSON",
        data=report_path.read_bytes(),
        file_name=report_path.name,
        mime="application/json",
        use_container_width=True,
        key="video_report_download",
    )
    st.caption(
        f"Frames: {summary['frames_processed']} | Objects: {summary['total_detections']} | Classes: {summary['unique_classes']} | Avg conf: {summary['avg_confidence']}"
    )


def init_state() -> None:
    if "image_result" not in st.session_state:
        st.session_state.image_result = None
    if "image_input_fp" not in st.session_state:
        st.session_state.image_input_fp = None
    if "video_result" not in st.session_state:
        st.session_state.video_result = None
    if "video_input_fp" not in st.session_state:
        st.session_state.video_input_fp = None


def main() -> None:
    st.set_page_config(page_title="YOLO Demo UI", page_icon="🎥", layout="wide")
    ensure_dirs()
    init_state()

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        .app-card { border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 14px 16px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("YOLO Seminar Demo UI")
    history_rows = load_history()

    with st.sidebar:
        st.header("Cấu hình")
        weights = st.text_input(
            "Weights (.pt)",
            value=str(DEFAULT_CUSTOM) if DEFAULT_CUSTOM.exists() else "yolo11n.pt",
            help="Nếu chưa train custom thì dùng yolo11n.pt / yolov8n.pt.",
        )
        imgsz = st.slider("imgsz", min_value=320, max_value=1280, value=640, step=32)
        conf = st.slider("conf", min_value=0.01, max_value=0.90, value=0.25, step=0.01)
        device = st.selectbox("device", options=["cpu", "0"], index=0, help="Chọn '0' nếu máy có CUDA GPU.")

        st.divider()
        st.caption("Tip: train dataset mẫu bằng `python tools/make_sample_dataset.py` rồi `python train_sample_shapes.py`.")
        if st.button("Refresh lịch sử"):
            st.rerun()

    try:
        model = load_model(weights)
    except Exception as exc:
        st.error(f"Không load được model: {exc}")
        st.stop()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Ảnh")
        upload_img = st.file_uploader("Upload ảnh", type=["jpg", "jpeg", "png", "bmp", "webp"], key="img")
        cam = st.camera_input("Hoặc chụp bằng camera (ảnh)", key="cam")

        source_name = None
        image_bytes = None
        image_fingerprint = None
        image_preview = None

        if cam is not None:
            source_name = "camera_image"
            image_bytes = cam.getvalue()
            image_fingerprint = fingerprint_bytes(image_bytes)
            img_arr = np.frombuffer(image_bytes, dtype=np.uint8)
            image_preview = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        elif upload_img is not None:
            source_name = upload_img.name
            image_bytes = upload_img.getvalue()
            image_fingerprint = fingerprint_bytes(image_bytes)
            tmp_path = save_uploaded_file(upload_img)
            image_preview = cv2.imread(str(tmp_path))

        if image_preview is not None:
            st.image(cv2.cvtColor(image_preview, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
            auto_detect = cam is not None and image_fingerprint != st.session_state.image_input_fp
            manual_detect = st.button("Chạy detection cho ảnh", use_container_width=True)
            if auto_detect or manual_detect:
                try:
                    annotated, out_path, report_path, summary = process_image(
                        model,
                        image_preview,
                        source_name=source_name or "image",
                        weights=weights,
                        imgsz=imgsz,
                        conf=conf,
                        device=device,
                    )
                except Exception as exc:
                    st.error(f"Lỗi xử lý ảnh: {exc}")
                else:
                    st.session_state.image_result = {
                        "input_fp": image_fingerprint,
                        "annotated": annotated,
                        "out_path": out_path,
                        "report_path": report_path,
                        "summary": summary,
                    }
                    st.session_state.image_input_fp = image_fingerprint

        image_result = st.session_state.image_result
        if image_result and image_result.get("input_fp") == image_fingerprint:
            show_image_result(
                "current",
                image_result["annotated"],
                image_result["out_path"],
                image_result["report_path"],
                image_result["summary"],
            )

    with col2:
        st.subheader("Video")
        upload_vid = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"], key="vid")

        if upload_vid is not None:
            video_bytes = upload_vid.getvalue()
            video_fingerprint = fingerprint_bytes(video_bytes)
            st.video(video_bytes)
            if st.button("Chạy detection cho video", use_container_width=True):
                try:
                    tmp_path = save_uploaded_file(upload_vid)
                    out, csv_path, report_path, summary = process_video(
                        model,
                        tmp_path,
                        source_name=upload_vid.name,
                        weights=weights,
                        imgsz=imgsz,
                        conf=conf,
                        device=device,
                    )
                except Exception as exc:
                    st.error(f"Lỗi xử lý video: {exc}")
                else:
                    st.session_state.video_result = {
                        "input_fp": video_fingerprint,
                        "out_path": out,
                        "csv_path": csv_path,
                        "report_path": report_path,
                        "summary": summary,
                    }
                    st.session_state.video_input_fp = video_fingerprint

        video_result = st.session_state.video_result
        if upload_vid is not None and video_result and video_result.get("input_fp") == fingerprint_bytes(upload_vid.getvalue()):
            show_video_result(
                video_result["out_path"],
                video_result["csv_path"],
                video_result["report_path"],
                video_result["summary"],
            )

    st.divider()
    st.subheader("Lịch sử detect gần đây")
    if history_rows:
        display_rows = []
        for row in reversed(history_rows[-20:]):
            display_rows.append(
                {
                    "time": row.get("time"),
                    "kind": row.get("kind"),
                    "source": row.get("source"),
                    "objects": row.get("total_detections"),
                    "classes": row.get("unique_classes"),
                    "avg_conf": row.get("avg_confidence"),
                    "max_conf": row.get("max_confidence"),
                    "notes": row.get("notes"),
                }
            )
        st.dataframe(display_rows, use_container_width=True, hide_index=True)
    else:
        st.info("Chưa có lịch sử detect nào được lưu.")


if __name__ == "__main__":
    main()
