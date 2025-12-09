# Deep Learning Final Project: Multi-Object Tracking & Re-Identification

**Authors:** Jim Huynh | Joseph Marinello | Ailing Nan | Kenny Phan
- **Lead Developer (Faster RCNN and Siamese Network):** _Ailing Nan_
- **Lead Developer (Yolo & Deepsort):** _Kenny Phan_
- **Code Reviewer and Github:** _Joseph Marinello_
- **Lead Document, Validation, and Presentation/Reporting:** _Jim Huynh_

## Project Overview

This project implements and compares multiple deep learning pipelines for multi-object detection and tracking (MOT) on pedestrians. We leverage the MOT16 benchmark for tracking and detection, and the Market-1501 dataset for training similarity models.

The project consists of three main components:

- **Object Detection:** Faster R-CNN with ResNet50 backbone.
- **Tracking Pipeline:** YOLOv8 coupled with DeepSort for real-time tracking.
- **Similarity/Re-ID:** A Siamese Network trained with Contrastive Loss for visual similarity.

## 1. Faster R-CNN Pipeline

**Notebook:** `Faster_RCNN.ipynb`

This component implements a two-stage object detector using PyTorch's `fasterrcnn_resnet50_fpn` model pre-trained on COCO.

### Key Features:

- **Data Handling:** Custom `MOT16TrainDataset` and `MOT16EvalDataset` classes to parse MOT16 ground truth and images.
- **Training:** Fine-tunes the Faster R-CNN head for 2 classes (background + pedestrian) using AdamW optimizer and a cosine learning rate scheduler.
- **Data Augmentation:** Implements custom augmentations including:
  - Random Horizontal Flip
  - Random Scale
  - Color Jitter
  - Random Gaussian Blur
  - CutOut
- **Inference:** Generates bounding box detections on test sequences and renders visualization videos.

## 2. YOLO + DeepSort Tracking Pipeline

**Notebook:** `YOLO_Deepsort.ipynb`

This pipeline combines a state-of-the-art one-stage detector (YOLOv8) with a tracking algorithm (DeepSort) to assign unique IDs to pedestrians across video frames.

### Key Features:

- **Data Preprocessing:** Converts MOT16 ground truth data into YOLO-compatible label format (normalized coordinates).
- **Detector Training:** Trains a YOLOv8-small (`yolov8s.pt`) model on the MOT16 training set.
- **Tracking:** Uses DeepSort to associate detections across frames based on Kalman filtering and visual appearance features.
- **Evaluation:** Computes standard MOT metrics using the `motmetrics` library:
  - MOTA (Multi-Object Tracking Accuracy)
  - IDF1 (ID F1 Score)
  - MOTP (Multi-Object Tracking Precision)
- **Visualization:** Outputs video files with bounding boxes and unique tracking IDs overlaid.

## 3. Siamese Similarity Model

**Notebook:** `SimilarityModel.ipynb`

This notebook implements a Siamese Neural Network designed to learn a similarity metric between two images, which is crucial for the Re-Identification (Re-ID) task in tracking.

### Key Features:

- **Dataset:** Uses the Market-1501 dataset, generating pairs of images (positive pairs for the same person, negative pairs for different people).
- **Architecture:** A custom Convolutional Neural Network (CNN) with three convolutional layers followed by fully connected layers.
- **Loss Function:** Optimized using Contrastive Loss, penalizing the distance between positive pairs and enforcing a margin between negative pairs.
- **Goal:** To output a similarity score that helps re-identify objects that re-enter a frame or are occluded.

## Setup & Requirements

### Datasets

- **MOT16:** Located at `/content/drive/MyDrive/Deep_Learning/Final Project/MOT16.zip`
- **Market-1501:** Located at `/content/drive/MyDrive/Deep_Learning/Final Project/Market_1501.zip`

### Dependencies

The following Python libraries are required:

- `torch` & `torchvision`
- `ultralytics` (for YOLOv8)
- `deep-sort-realtime`
- `motmetrics` (for evaluation)
- `opencv-python` (cv2)
- `pandas`, `numpy`, `matplotlib`
- `albumentations`

## Usage

1. Ensure the datasets are uploaded to the specified Google Drive paths.
2. Mount Google Drive in the Colab notebooks.
3. Run the notebooks in the following order (independent pipelines):
   - Run `YOLO_Deepsort.ipynb` for full tracking and metric evaluation.
   - Run `Faster_RCNN.ipynb` for detection baselines.
   - Run `SimilarityModel.ipynb` to train the Re-ID feature extractor.
