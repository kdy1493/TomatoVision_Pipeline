# 🍅 TomatoVision Pipeline

스마트팜 토마토 수확 로봇을 위한 **“2D 기반 3D 인식 파이프라인(2D-centric 3D Perception Pipeline)”** 구현 레포지토리입니다.  
YOLO(Detection) · FastSAM(Segmentation) · FoundationStereo(Depth Estimation) · ZED ROS2 Bag Processing · Rerun 3D Visualization을 하나의 통합 흐름으로 구성하였습니다. 

본 파이프라인은 토마토 객체의 2D 탐지 및 인스턴스 분할 결과를 기반으로 깊이 정보를 결합하여, **3D Point Cloud 생성 → 3D 위치 계산 → 시각화**까지 한 번에 수행합니다.

![Pipeline Overview](docs/pipeline_diagram.png)

*전체 파이프라인 구조: Stereo Vision → Depth Estimation → 2D Detection → Instance Segmentation → 3D Reconstruction*

---

## ⚡ Quick Start: 3D Visualization Pipeline

이 레포지토리의 핵심 기능은 **전처리된 RGB/Depth 시퀀스를 기반으로 YOLO + FastSAM + Depth 정보를 결합하여 3D 공간상에서 토마토를 정밀하게 시각화하는 기능**입니다.  

전처리가 완료된 데이터 디렉토리를 입력하면, 아래 요소들이 통합된 3D 뷰어가 자동 실행됩니다:

- RGB / Depth 프레임 스트림
- 카메라 Pose & Trajectory
- YOLO 기반 토마토 검색 및 성숙도 분류
- FastSAM 기반 인스턴스 마스크
- Masked Point Cloud 생성 및 시각화
- 토마토 3D centroid 및 Bounding Box 추정 결과

### ▶️ 실행 명령

```bash
python scripts/3d_visualizer.py data/tomato_data/tomato_video/smartfarm_251114
```

**실행 즉시 Rerun GUI가 열리며**, 프레임 단위 RGB/Depth/Mask/PointCloud 및 3D 위치 정보가 시간 축에 따라 시각화됩니다.

---

## 🔧 Installation

### Requirements
- Python 3.11+
- CUDA 12.1 (GPU 사용 시)
- uv 패키지 매니저

### Setup
```bash
git clone <repository-url>
cd tomatovision_pipeline
uv sync
```

가상환경 활성화:
```bash
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

---

## 📘 Usage

### 1. Dataset Download & Preprocessing — Laboro-Tomato
```bash
python scripts/download_dataset.py
```

자동 수행:
- Kaggle 다운로드
- 클래스 ID 6→3 재매핑
- COCO/YOLO 데이터 생성
- data.yaml 자동 생성

> 기본 제공된 `weight/trained_yolo12n.pt` 사용을 권장하므로, YOLO 학습은 선택 사항입니다.

---

### 2. YOLO Training / Prediction

학습:
```bash
python scripts/yolo.py --config configs/yolo.yaml --mode train
```

---
예측:
```bash
python scripts/yolo.py --config configs/yolo.yaml --mode predict
```

## 📊 Dataset Viewer (FiftyOne)

```bash
python scripts/fiftyone_viewer.py --config configs/fiftyone_viewer.yaml
```
---

### 3. ROS2 Bag Preprocessing (ZED → RGB/Depth/Trajectory)

```bash
python scripts/rosbag_preprocess.py <bag_path> --output-dir <output>
```

옵션:
- `--depth-source foundation`
- `--depth-source zed`
- `--depth-source both`

결과 구조:
```
output/
├── rgb.mp4
├── depth_foundation.mp4
├── trajectory.json
├── pointcloud.json
└── meta.json
```

---

## 🎥 Video Segmentation (YOLO + FastSAM)

비디오 파일에서 토마토만 추출하여 이진 마스크 영상을 생성합니다. 

```bash
python scripts/video_mask_segmenter_demo.py --config configs/video_masksegment_pipeline.yaml
```

출력:
- `rgb_masked.mp4` - 토마토 영역만 흰색으로 표시된 마스크 영상
- `rgb_masked_yolo.mp4` - YOLO 탐지 결과가 오버레이된 영상

---

## 📁 Project Structure

```
tomatovision_pipeline/
├── scripts/
│   ├── 3d_visualizer.py          # ⭐ Main Pipeline
│   ├── rosbag_preprocess.py
│   ├── yolo.py
│   ├── video_mask_segmenter_demo.py
│   └── fiftyone_viewer.py
├── src/
│   ├── model/
│   ├── preprocess/
│   └── viewer/
├── configs/
├── data/
├── weight/
└── notebook/
```

---

## ⚙ Configuration Files

- **configs/sensor_config.yaml** : segmentation, depth, rerun 옵션
- **configs/yolo.yaml** : YOLO 학습/추론 설정
- **configs/video_masksegment_pipeline.yaml** : 영상 마스킹 파이프라인
- **configs/fiftyone_viewer.yaml** : COCO/YOLO 데이터셋 로딩

---

