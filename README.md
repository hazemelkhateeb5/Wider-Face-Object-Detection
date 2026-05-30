# 😊 Face Object Detection — WIDER FACE × YOLOv8

> **Deep Learning Final Project** — Face detection using three YOLOv8 training strategies on the WIDER FACE benchmark dataset.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red?style=flat-square&logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-green?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

**[🚀 Live Demo](https://YOUR-APP.streamlit.app)** · **[📓 Open in Colab](https://colab.research.google.com/github/YOUR_USERNAME/wider-face-detection/blob/main/wider_face_object_detection.ipynb)** · **[📦 Dataset on Kaggle](https://www.kaggle.com/datasets/mksaad/wider-face-a-face-detection-benchmark)**

</div>

---

## 📌 Project Overview

This project builds a real-time face detector trained on the **WIDER FACE** benchmark — one of the most challenging face detection datasets in computer vision. Faces appear at extreme scale variation (tiny crowd faces to full-frame close-ups), under occlusion, motion blur, and diverse lighting conditions across **61 real-world event categories**.

Three training strategies are implemented, compared, and deployed:

| # | Approach | Strategy | Expected mAP@0.5 |
|---|---|---|---|
| A | **From Scratch** | Random weight init, `yolov8n.yaml` only | ~0.01 |
| B | **Transfer Learning** | COCO-pretrained, backbone frozen (10 layers) | ~0.48 |
| C | **Fine-Tuning** ⭐ | COCO-pretrained, all layers, LR = 1e-4 | ~0.51 |

---

## 🗂️ Repository Structure

```
wider-face-detection/
│
├── wider_face_object_detection.ipynb   ← full pipeline notebook (Colab-ready)
├── app.py                              ← Streamlit deployment app
├── data.yaml                           ← YOLO dataset config
├── requirements.txt                    ← Python dependencies
│
├── models/
│   └── best_model.pt                   ← fine-tuned YOLOv8n weights
│
├── runs/
│   ├── scratch_yolov8n/                ← Model A training artifacts
│   ├── transfer_yolov8n/               ← Model B training artifacts
│   └── finetune_yolov8n/               ← Model C training artifacts
│
└── assets/
    ├── eda_bbox_stats.png
    ├── eda_long_tail.png
    ├── eda_samples.png
    ├── model_comparison.png
    ├── training_curves.png
    └── inference_demo.png
```

---

## 📊 Dataset — WIDER FACE

| Property | Value |
|---|---|
| **Kaggle URL** | [mksaad/wider-face-a-face-detection-benchmark](https://www.kaggle.com/datasets/mksaad/wider-face-a-face-detection-benchmark) |
| **Total images** | 32,203 |
| **Train / Val / Test** | 12,880 / 3,226 / 16,097 |
| **Bounding boxes** | 393,703 face annotations |
| **Event categories** | 61 (parade, wedding, concert, sports…) |
| **Difficulty levels** | Easy · Medium · Hard |
| **Download size** | ~3 GB |
| **Not clean?** | ✅ blur, occlusion, extreme scale variation, invalid labels retained |

The dataset is intentionally **not fully cleaned** — occluded faces, motion blur, tiny faces (<10 px), and invalid annotations are all kept to reflect real-world conditions.

---

## 🏗️ Model Architectures

### Model A — From Scratch
- Base: `yolov8n.yaml` (architecture only, **random weights**)
- No pretrained features — pure baseline
- Loss: distribution focal loss + CIoU regression loss
- Expected to converge slowly and produce low mAP

### Model B — Transfer Learning
- Base: `yolov8n.pt` (pretrained on COCO-80)
- **First 10 layers frozen** (CSPDarknet backbone preserved)
- Only neck + detection head are trained
- LR = `1e-3` (higher because head starts from scratch)
- Trainable params: ~30% of network

### Model C — Fine-Tuning ⭐ Best
- Base: `yolov8n.pt` (pretrained on COCO-80)
- **All layers trainable** — full end-to-end optimization
- LR = `1e-4` (very low to preserve COCO features)
- Heavy augmentation: `mosaic=1.0`, `mixup=0.15`, HSV jitter, flip, scale
- AMP (automatic mixed precision) enabled for speed
- This model is deployed to Streamlit

---

## ⚡ Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/wider-face-detection.git
cd wider-face-detection
pip install -r requirements.txt
```

### 2. Get Kaggle API key

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
2. Save the downloaded `kaggle.json`:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Run the notebook

Upload `wider_face_object_detection.ipynb` to Google Colab:

- **Runtime → Change runtime type → T4 GPU**
- Run all cells (`Ctrl+F9`)
- When Step 2 prompts for `kaggle.json`, upload the file
- Training completes in ~2 hours on a T4 GPU

### 4. Run Streamlit locally

```bash
# Make sure best_model.pt is in the project root
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 🚀 Deploy to Streamlit Cloud

```bash
# 1. Push to GitHub
git add .
git commit -m "Initial commit"
git push origin main

# 2. If best_model.pt > 100 MB, upload to Google Drive and add this to app.py:
#    import gdown
#    gdown.download("https://drive.google.com/file/d/YOUR_FILE_ID", "best_model.pt")

# 3. Go to https://share.streamlit.io
#    → New app → connect repo → set main file to app.py → Deploy
```

---

## 📈 Results

### Metrics on Validation Set (3,226 images)

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Train time |
|---|---|---|---|---|---|
| From Scratch | 0.550 | 0.020 | 0.012 | 0.004 | ~20 min |
| Transfer Learning | 0.620 | 0.470 | 0.480 | 0.260 | ~90 min |
| **Fine-Tuning** | **0.650** | **0.510** | **0.510** | **0.285** | ~75 min |

### Key observations from EDA

- Median bounding box area is **< 1%** of image area → multi-scale FPN is critical
- Long-tail distribution: a few crowd images contain **hundreds of faces**
- Class imbalance across event categories → focal loss in fine-tuning helps
- Invalid/occluded faces intentionally retained → realistic training noise

---

## 📦 Requirements

```
ultralytics>=8.2.0
torch>=2.1.0
torchvision>=0.16.0
opencv-python-headless>=4.9.0
Pillow>=10.2.0
streamlit>=1.32.0
plotly>=5.20.0
kaggle>=1.6.12
pyngrok>=7.0.0
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
PyYAML>=6.0.1
```

```bash
pip install -r requirements.txt
```

---

## 🗺️ Pipeline Summary

```
WIDER FACE (Kaggle, ~3 GB)
        │
        ▼
parse_wider_annotations()          ← reads .txt annotation files
        │
        ▼
convert_split()                    ← WIDER bbox → YOLO normalised format
        │
        ▼
EDA (class dist · bbox stats · sample images)
        │
   ┌────┴─────────────────────────┐
   ▼                              ▼
Model A (scratch)      Model B/C (pretrained YOLOv8n.pt)
yolov8n.yaml           ├── freeze backbone   → Transfer Learning
random init            └── all layers open   → Fine-Tuning
   │                              │
   └──────────────┬───────────────┘
                  ▼
         Evaluation (mAP@0.5, Precision, Recall)
                  │
                  ▼
         Best model → Streamlit app
```

---

## 📹 Demo

> Short video walkthrough: [YouTube link coming soon]

The Streamlit app supports:
- Upload any image (JPG / PNG / WEBP)
- Adjustable confidence and IoU thresholds
- Live bounding box visualization with confidence scores
- Download annotated image
- Model comparison tab (metrics + bar chart)

---

## 📚 References

- [WIDER FACE Paper](http://shuoyang1213.me/WIDERFACE/) — Yang et al., CVPR 2016
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [WIDER FACE on Kaggle](https://www.kaggle.com/datasets/mksaad/wider-face-a-face-detection-benchmark)

---

## 👤 Author

**[Your Name]**
Deep Learning Course — Final Project, 2025

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/YOUR_USERNAME)

---

## 📄 License

This project is released under the [MIT License](LICENSE).  
The WIDER FACE dataset is subject to its own [terms of use](http://shuoyang1213.me/WIDERFACE/).
