# Computer Vision YOLOv11 Seminar Project

This repository contains a small end-to-end demo for object detection with Ultralytics YOLO:

- `slides.md`: seminar slide content with speaker notes.
- `demo_yolo_ultralytics.py`: CLI demo for webcam, video, or image input.
- `app_streamlit.py`: Streamlit UI for upload-and-detect workflows.
- `tools/make_sample_dataset.py`: synthetic YOLO dataset generator.
- `train_sample_shapes.py`: quick training script for the sample dataset.
- `export_weights.py`: export trained weights to ONNX or other formats.

Included assets:

- `datasets/sample_shapes/`: tiny synthetic dataset for a fast demo.
- `runs/sample_shapes_ft/weights/best.pt`: latest custom weights trained on the sample dataset.
- `yolo11n.pt`: fallback pretrained weights used when no custom model is available.

## Run

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Generate the sample dataset and train:

```powershell
python tools\make_sample_dataset.py
python train_sample_shapes.py --epochs 10 --device cpu
```

Run the demos:

```powershell
python demo_yolo_ultralytics.py --show
streamlit run app_streamlit.py
```

## Notes

- The default custom weights path is `runs/sample_shapes_ft/weights/best.pt`.
- Runtime files such as logs, cache, `outputs/`, and local virtual environments are intentionally ignored by Git.
- If `yolo11n.pt` is missing, Ultralytics can download it automatically when the demo starts.
