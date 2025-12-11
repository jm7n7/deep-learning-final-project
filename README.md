# Deep Learning Final Project: Multi-Object Tracking & Re-Identification

**Authors:** Jim Huynh | Joseph Marinello | Ailing Nan | Kenny Phan
#### Project Manager: _Kenny Phan_
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

## 4. Tracker Pipeline (F-RCNN + Siamese)

**Notebook:** `tracker.ipynb`

This notebook implements a complete multi-object tracking pipeline that combines Faster R-CNN for person detection with a Siamese network for person re-identification. This approach provides robust tracking by leveraging both spatial detection and visual appearance features.

### Key Features:

- **Detection Stage:** Uses a fine-tuned Faster R-CNN model to detect persons in each frame with confidence thresholding (default: 0.7).
- **Embedding Extraction:** For each detected person, crops the bounding box region, resizes it to 128x64, and extracts a 256-dimensional embedding using the Siamese network.
- **Data Association:** Matches detections to existing tracks using cosine similarity between embeddings:
  - Maintains a sliding window of recent embeddings (default: 5 frames) per track for robust matching
  - Uses average similarity across multiple historical embeddings to handle appearance changes
  - Associates detections to tracks if similarity exceeds threshold (default: 0.6)
- **Track Management:**
  - Creates new tracks for unmatched detections (assumed to be new persons entering the scene)
  - Updates matched tracks with new embeddings and positions
  - Removes stale tracks that haven't been seen for a specified number of frames (default: 30 frames)
- **Output Generation:**
  - Saves tracking results in MOTChallenge format (frame, id, x, y, w, h, conf, ...)
  - Generates annotated video with colored bounding boxes and track IDs
  - Computes evaluation metrics (MOTA, MOTP, IDF1, etc.) if ground truth is available

### Architecture Components:

- **Siamese Network:** Custom CNN architecture with 3 convolutional layers and 2 fully connected layers, trained on Market-1501 dataset for person re-identification.
- **Faster R-CNN Detector:** Pre-trained on ImageNet and fine-tuned on MOT16 training data for person detection.
- **MOT16 Dataset Classes:** 
  - `MOT16TrainDataset`: Loads training sequences with ground truth annotations
  - `MOT16TestDataset`: Loads test sequences without annotations for inference

### Configuration Parameters:

- `BBOX_SCORE_THRESH`: Minimum confidence for detections (default: 0.7)
- `MAX_EMBEDDING_HISTORY`: Number of recent embeddings per track (default: 5)
- `MAX_FRAMES_MISSING`: Maximum frames before track deletion (default: 30)
- `SIMILARITY_THRESHOLD`: Minimum cosine similarity for association (default: 0.6)

### Output Files:

- **Tracking Results:** `results_FasterRCNN/FasterRCNN_tracker_results.txt` (MOT format)
- **Visualization Video:** `results_FasterRCNN/tracked_test_video.mp4`
- **Evaluation Metrics:** `results_FasterRCNN/FasterRCNN_tracker_metrics.csv` (if ground truth available)
- **Annotated Frames:** `out/` directory with individual frame images

### Usage:

1. Ensure both the Faster R-CNN detector model (`models_FasterRCNN/bbox_detector.pth`) and Siamese network (`siamese_network.pth`) are trained and saved.
2. Update `PROJECT_PATH` to match your Google Drive directory structure.
3. Configure paths for model files, test sequence, and output directories in the configuration cell.
4. Run all cells sequentially to execute the complete tracking pipeline.

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
3. Run the notebooks in the following order:
   
   **Training Phase:**
   - Run `Faster_RCNN.ipynb` to train the person detector (or use pre-trained weights).
   - Run `SimilarityModel.ipynb` to train the Siamese network for person re-identification.
   
   **Tracking Phase (Independent Pipelines):**
   - Run `YOLO_Deepsort.ipynb` for YOLOv8 + DeepSort tracking pipeline.
   - Run `tracker.ipynb` for Faster R-CNN + Siamese network tracking pipeline (requires trained models from above).
