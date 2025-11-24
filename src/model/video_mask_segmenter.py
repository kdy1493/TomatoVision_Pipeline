"""
VideoMaskSegmenter 모듈

YOLO 객체 탐지 + FastSAM 세그멘테이션을 결합한 동적 마스킹 클래스
video_mask_segmenter_demo.py의 핵심 로직을 재사용하며, ImageSegmenter와 호환되는 인터페이스 제공

사용 예:
    segmenter = VideoMaskSegmenter(
        yolo_model_path="weight/yolo12n.pt",
        fastsam_model_path="weight/FastSAM-s.pt"
    )
    mask = segmenter.get_mask(rgb_image)  # (H, W) 이진 마스크 반환
"""

import numpy as np
import torch
import cv2
from ultralytics import YOLO, FastSAM
from pathlib import Path


class VideoMaskSegmenter:
    """
    YOLO + FastSAM을 이용한 이미지 세그멘테이션
    
    동작 흐름:
    1. YOLO로 객체 탐지 (바운딩박스 추출)
    2. 감지된 박스 영역 내에서 FastSAM으로 정밀한 세그멘테이션
    3. 여러 마스크를 하나로 합쳐 최종 이진 마스크 생성
    """
    
    def __init__(self, 
                 yolo_model_path: str = "weight/trained_yolo12n.pt",
                 fastsam_model_path: str = "weight/FastSAM-s.pt",
                 device: str = None,
                 yolo_confidence_threshold: float = 0.25,
                 yolo_input_size: int = 640):
        """
        VideoMaskSegmenter 초기화
        
        Args:
            yolo_model_path (str): YOLO 모델 파일 경로
                                   예: "weight/trained_yolo12n.pt"
            fastsam_model_path (str): FastSAM 모델 파일 경로
                                      예: "weight/FastSAM-s.pt"
            device (str): 실행 디바이스 ('cuda', 'cpu', 또는 None)
                         None이면 GPU 자동 선택 (사용 가능 시)
            yolo_confidence_threshold (float): YOLO 신뢰도 임계값 (0.0~1.0)
                                               값이 클수록 정확한 탐지만 선택
            yolo_input_size (int): YOLO 입력 이미지 크기 (정사각형)
                                  예: 640, 1024 (클수록 정확하나 느림)
        """
        # 디바이스 설정
        if device is None:
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.yolo_confidence_threshold = yolo_confidence_threshold
        self.yolo_input_size = yolo_input_size
        
        # YOLO 모델 로드
        print(f"[VideoMaskSegmenter] YOLO 모델 로드 중: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        try:
            self.yolo_model.to(self.device)
        except Exception as e:
            print(f"[경고] YOLO 디바이스 설정 실패: {e}")
        
        # FastSAM 모델 로드
        print(f"[VideoMaskSegmenter] FastSAM 모델 로드 중: {fastsam_model_path}")
        self.fastsam_model = FastSAM(fastsam_model_path)
        
        print(f"[VideoMaskSegmenter] 초기화 완료 (디바이스: {self.device})")
    
    def get_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        RGB 이미지에서 세그멘테이션 마스크 생성
        
        Args:
            image_rgb (np.ndarray): RGB 이미지 배열
                                   형태: (높이, 너비, 3)
                                   값 범위: 0~255 (uint8) 또는 0~1 (float32)
        
        Returns:
            np.ndarray: 이진 마스크 배열
                       형태: (높이, 너비)
                       값: True(전경)/False(배경)
                       
        처리 과정:
            1. YOLO 탐지: 객체 바운딩박스 추출
            2. FastSAM 분할: 각 박스 내 정밀한 마스크 생성
            3. 마스크 통합: 모든 마스크를 OR 연산으로 합치기
            4. 리사이징: 원본 이미지 크기로 조정
        """
        image_height, image_width = image_rgb.shape[:2]
        
        # OpenCV는 BGR 사용, YOLO도 BGR 입력 권장
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # ─────────────────────────────────────────────────────────
        # Step 1: YOLO로 객체 탐지
        # ─────────────────────────────────────────────────────────
        yolo_detection_results = self.yolo_model(
            image_bgr, 
            conf=self.yolo_confidence_threshold, 
            imgsz=self.yolo_input_size, 
            device=self.device, 
            verbose=False
        )
        yolo_result = yolo_detection_results[0]
        
        # 초기 마스크: 모두 배경(False)
        binary_segmentation_mask = np.zeros(
            (image_height, image_width), 
            dtype=bool
        )
        
        # ─────────────────────────────────────────────────────────
        # Step 2: 탐지된 박스가 있으면 FastSAM 실행
        # ─────────────────────────────────────────────────────────
        if yolo_result.boxes is not None and len(yolo_result.boxes) > 0:
            # 모든 박스의 좌표 추출
            detection_boxes = []
            for detected_box in yolo_result.boxes:
                detection_boxes.append(detected_box.xyxy[0])
            
            if len(detection_boxes) > 0:
                # 박스 좌표를 텐서로 변환
                box_coordinates_tensor = torch.stack(detection_boxes)
                
                # ─────────────────────────────────────────────────────────
                # Step 3: FastSAM으로 정밀한 세그멘테이션
                # ─────────────────────────────────────────────────────────
                fastsam_segmentation_results = self.fastsam_model.predict(
                    source=image_rgb,  # FastSAM은 RGB 입력
                    bboxes=box_coordinates_tensor,
                    device=self.device,
                    retina_masks=True,  # 고해상도 마스크 사용
                    imgsz=self.yolo_input_size,
                    conf=0.5,
                    iou=0.9,  # NMS IOU 임계값
                    verbose=False
                )
                
                # ─────────────────────────────────────────────────────────
                # Step 4: FastSAM 마스크 추출 및 통합
                # ─────────────────────────────────────────────────────────
                if (len(fastsam_segmentation_results) > 0 and 
                    getattr(fastsam_segmentation_results[0], "masks", None) is not None):
                    
                    # 마스크 데이터 추출
                    raw_masks = fastsam_segmentation_results[0].masks.data
                    
                    # PyTorch 텐서 → NumPy 배열로 변환
                    if torch.is_tensor(raw_masks):
                        masks_array = raw_masks.detach().cpu().numpy()
                    else:
                        masks_array = np.asarray(raw_masks)
                    
                    # 여러 마스크를 하나로 통합 (OR 연산)
                    if masks_array.ndim >= 2:
                        # 3차원 이상: 여러 마스크 존재
                        combined_mask = np.any(masks_array > 0, axis=0)
                    else:
                        # 1차원 또는 2차원: 단일 마스크
                        combined_mask = masks_array > 0
                    
                    # ─────────────────────────────────────────────────────────
                    # Step 5: 원본 이미지 크기로 리사이징
                    # ─────────────────────────────────────────────────────────
                    if combined_mask.shape != (image_height, image_width):
                        resized_mask = cv2.resize(
                            combined_mask.astype(np.uint8),
                            (image_width, image_height),
                            interpolation=cv2.INTER_NEAREST
                        )
                        binary_segmentation_mask = resized_mask.astype(bool)
                    else:
                        binary_segmentation_mask = combined_mask
        
        return binary_segmentation_mask
    
    def get_mask_with_confidence(self, image_rgb: np.ndarray) -> tuple:
        """
        RGB 이미지에서 마스크와 신뢰도 정보 반환 (디버깅용)
        
        Args:
            image_rgb (np.ndarray): RGB 이미지 배열
        
        Returns:
            tuple: (binary_mask, detection_count, processing_info)
                - binary_mask: (H, W) 이진 마스크
                - detection_count: 탐지된 객체 수
                - processing_info: 처리 정보 딕셔너리
        """
        image_height, image_width = image_rgb.shape[:2]
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # YOLO 탐지
        yolo_results = self.yolo_model(
            image_bgr,
            conf=self.yolo_confidence_threshold,
            imgsz=self.yolo_input_size,
            device=self.device,
            verbose=False
        )
        yolo_result = yolo_results[0]
        
        detection_count = 0
        info = {
            "yolo_detections": 0,
            "fastsam_masks": 0,
            "mask_area_ratio": 0.0
        }
        
        binary_mask = np.zeros((image_height, image_width), dtype=bool)
        
        if yolo_result.boxes is not None and len(yolo_result.boxes) > 0:
            detection_count = len(yolo_result.boxes)
            info["yolo_detections"] = detection_count
            
            boxes = [box.xyxy[0] for box in yolo_result.boxes]
            box_tensor = torch.stack(boxes)
            
            fastsam_results = self.fastsam_model.predict(
                source=image_rgb,
                bboxes=box_tensor,
                device=self.device,
                retina_masks=True,
                imgsz=self.yolo_input_size,
                conf=0.5,
                iou=0.9,
                verbose=False
            )
            
            if len(fastsam_results) > 0 and getattr(fastsam_results[0], "masks", None) is not None:
                masks = fastsam_results[0].masks.data
                if torch.is_tensor(masks):
                    masks = masks.detach().cpu().numpy()
                else:
                    masks = np.asarray(masks)
                
                if masks.ndim >= 2:
                    combined = np.any(masks > 0, axis=0)
                else:
                    combined = masks > 0
                
                info["fastsam_masks"] = len(masks) if masks.ndim >= 2 else 1
                
                if combined.shape != (image_height, image_width):
                    resized = cv2.resize(
                        combined.astype(np.uint8),
                        (image_width, image_height),
                        interpolation=cv2.INTER_NEAREST
                    )
                    binary_mask = resized.astype(bool)
                else:
                    binary_mask = combined
                
                mask_pixels = np.sum(binary_mask)
                total_pixels = image_height * image_width
                info["mask_area_ratio"] = mask_pixels / total_pixels if total_pixels > 0 else 0.0
        
        return binary_mask, detection_count, info


class DummySegmenter:
    """
    세그멘테이션 비활성화 시 사용되는 더미 세그멘터
    항상 전체 이미지를 전경으로 반환
    """
    def get_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        전체 이미지를 전경(True)으로 반환
        
        Args:
            image_rgb (np.ndarray): RGB 이미지
        
        Returns:
            np.ndarray: 모두 True인 마스크 (H, W)
        """
        return np.ones(
            (image_rgb.shape[0], image_rgb.shape[1]),
            dtype=bool
        )
