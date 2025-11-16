"""
3D 바운딩박스 시각화 모듈

Rerun에서 3D 공간에 바운딩박스와 라벨을 표시하는 기능 제공
카메라 프리즘의 깊이 정보를 이용하여 2D 탐지를 3D 바운딩박스로 변환

사용 예:
    from src.viewer.bbox_3d_visualizer import BBox3DVisualizer
    
    visualizer = BBox3DVisualizer(
        fx=500.0, fy=500.0, cx=640, cy=360,
        depth_min=0.4, depth_max=1.0
    )
    visualizer.log_3d_bboxes_from_2d_detections(
        detections=[(x1, y1, x2, y2, class_id, confidence), ...],
        depth_image=depth_map,
        class_names={0: "tomato", 1: "leaf", ...}
    )
"""

import numpy as np
import torch
import rerun as rr
from typing import List, Tuple, Dict, Optional


class BBox3DVisualizer:
    """
    2D 탐지 결과를 3D 바운딩박스로 변환하여 Rerun에 시각화
    """
    
    def __init__(self,
                 fx: float,
                 fy: float,
                 cx: float,
                 cy: float,
                 depth_min: float = 0.4,
                 depth_max: float = 1.0):
        """
        초기화
        
        Args:
            fx, fy: 카메라 초점거리 (픽셀 단위)
            cx, cy: 카메라 주점 (픽셀 단위)
            depth_min: 최소 깊이 (m)
            depth_max: 최대 깊이 (m)
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_min = depth_min
        self.depth_max = depth_max
    
    def _pixel_to_3d(self, 
                     x_px: float, 
                     y_px: float, 
                     depth_m: float) -> np.ndarray:
        """
        2D 픽셀 좌표를 3D 카메라 좌표계로 변환
        
        카메라 좌표계: X(오른쪽), Y(아래), Z(앞)
        
        Args:
            x_px, y_px: 픽셀 좌표
            depth_m: 깊이 (미터)
        
        Returns:
            np.ndarray: 3D 카메라 좌표 [X, Y, Z]
        """
        x_cam = (x_px - self.cx) * depth_m / self.fx
        y_cam = (y_px - self.cy) * depth_m / self.fy
        z_cam = depth_m
        return np.array([x_cam, y_cam, z_cam])
    
    def _get_bbox_3d_corners(self,
                             bbox_2d: Tuple[float, float, float, float],
                             depth_image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        2D 바운딩박스의 4개 코너를 깊이 이미지로부터 3D 좌표로 변환
        
        Args:
            bbox_2d: (x1, y1, x2, y2) 픽셀 좌표
            depth_image: 깊이 맵 (H, W)
        
        Returns:
            Tuple[corners_3d, mean_depth]
                - corners_3d: (4, 3) 3D 코너 좌표 배열
                - mean_depth: 평균 깊이
        """
        x1, y1, x2, y2 = bbox_2d
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 이미지 경계 처리
        h, w = depth_image.shape
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        
        # 바운딩박스 영역의 깊이 샘플링 (평균)
        roi_depth = depth_image[y1:y2+1, x1:x2+1]
        valid_depths = roi_depth[(roi_depth > self.depth_min) & (roi_depth < self.depth_max)]
        
        if len(valid_depths) == 0:
            # 유효한 깊이가 없으면 기본값 사용
            mean_depth = (self.depth_min + self.depth_max) / 2
        else:
            mean_depth = float(np.mean(valid_depths))
        
        # 4개 코너 픽셀 좌표
        corners_px = np.array([
            [x1, y1],  # 좌상단
            [x2, y1],  # 우상단
            [x1, y2],  # 좌하단
            [x2, y2],  # 우하단
        ], dtype=float)
        
        # 각 코너를 3D로 변환
        corners_3d = np.array([
            self._pixel_to_3d(px[0], px[1], mean_depth)
            for px in corners_px
        ])
        
        return corners_3d, mean_depth
    
    def _draw_bbox_edges(self,
                         corners_3d: np.ndarray,
                         entity_path: str,
                         color: Tuple[int, int, int] = (0, 255, 0),
                         radius: float = 0.005):
        """
        3D 바운딩박스의 엣지를 라인으로 그리기
        
        Args:
            corners_3d: (4, 3) 코너 좌표
            entity_path: Rerun 엔티티 경로
            color: RGB 색상 (0~255)
            radius: 라인 반지름
        """
        # 바운딩박스 에지 연결 (2D 직사각형이므로 4개 에지)
        edges = [
            (0, 1),  # 좌상단 - 우상단
            (1, 3),  # 우상단 - 우하단
            (3, 2),  # 우하단 - 좌하단
            (2, 0),  # 좌하단 - 좌상단
        ]
        
        for start, end in edges:
            start_point = corners_3d[start]
            end_point = corners_3d[end]
            
            rr.log(
                entity_path,
                rr.LineStrips3D(
                    [[start_point.tolist(), end_point.tolist()]],
                    colors=[color],
                    radii=[radius]
                )
            )
    
    def log_3d_bboxes_from_yolo_detections(self,
                                           yolo_results,
                                           depth_image: np.ndarray,
                                           class_names: Optional[Dict[int, str]] = None,
                                           confidence_threshold: float = 0.5,
                                           show_labels: bool = True):
        """
        YOLO 탐지 결과로부터 3D 바운딩박스와 라벨 로깅
        
        Args:
            yolo_results: YOLO 모델의 결과 객체 (results[0])
            depth_image: 깊이 맵 (H, W)
            class_names: {클래스_id: 클래스_이름} 딕셔너리
            confidence_threshold: 신뢰도 필터링 임계값
            show_labels: 라벨 표시 여부
        """
        if (yolo_results.boxes is None or 
            len(yolo_results.boxes) == 0):
            return
        
        class_names = class_names or {}
        colors_map = self._get_color_palette(len(class_names))
        
        detection_idx = 0
        for box in yolo_results.boxes:
            # 신뢰도 확인
            conf = float(box.conf[0]) if hasattr(box, "conf") else 1.0
            if conf < confidence_threshold:
                continue
            
            # 클래스 정보
            class_id = int(box.cls[0]) if hasattr(box, "cls") else 0
            class_name = class_names.get(class_id, f"Class_{class_id}")
            
            # 2D 바운딩박스 좌표
            bbox_2d = box.xyxy[0].detach().cpu().numpy()
            
            try:
                # 3D 코너 계산
                corners_3d, mean_depth = self._get_bbox_3d_corners(
                    bbox_2d, depth_image
                )
                
                # 색상 선택
                color = colors_map.get(class_id, (0, 255, 0))
                
                # 엣지 그리기
                entity_path = f"world/detections/bbox_3d_{detection_idx}"
                self._draw_bbox_edges(
                    corners_3d,
                    entity_path,
                    color=color,
                    radius=0.003
                )
                
                # 라벨 표시
                if show_labels:
                    # 바운딩박스 중심에 라벨 표시
                    center_3d = corners_3d.mean(axis=0)
                    label_text = f"{class_name}\n{conf:.2f}"
                    
                    rr.log(
                        f"world/detections/label_{detection_idx}",
                        rr.TextLog(label_text)
                    )
                    
                    # 중심점 표시
                    rr.log(
                        f"world/detections/center_{detection_idx}",
                        rr.Points3D(
                            [center_3d],
                            colors=[color],
                            radii=[0.01]
                        )
                    )
                
                detection_idx += 1
                
            except Exception as e:
                print(f"[경고] 3D 바운딩박스 계산 실패: {e}")
                continue
    
    def log_3d_bboxes_from_2d_detections(self,
                                         detections: List[Tuple],
                                         depth_image: np.ndarray,
                                         class_names: Optional[Dict[int, str]] = None,
                                         show_labels: bool = True):
        """
        수동 2D 탐지로부터 3D 바운딩박스 로깅
        
        Args:
            detections: [(x1, y1, x2, y2, class_id, confidence), ...] 리스트
            depth_image: 깊이 맵 (H, W)
            class_names: {클래스_id: 클래스_이름} 딕셔너리
            show_labels: 라벨 표시 여부
        """
        class_names = class_names or {}
        colors_map = self._get_color_palette(len(class_names))
        
        for idx, detection in enumerate(detections):
            x1, y1, x2, y2, class_id, conf = detection
            bbox_2d = (x1, y1, x2, y2)
            class_name = class_names.get(class_id, f"Class_{class_id}")
            color = colors_map.get(class_id, (0, 255, 0))
            
            try:
                corners_3d, mean_depth = self._get_bbox_3d_corners(
                    bbox_2d, depth_image
                )
                
                entity_path = f"world/detections/bbox_3d_{idx}"
                self._draw_bbox_edges(
                    corners_3d,
                    entity_path,
                    color=color,
                    radius=0.003
                )
                
                if show_labels:
                    center_3d = corners_3d.mean(axis=0)
                    label_text = f"{class_name}\n{conf:.2f}"
                    
                    rr.log(
                        f"world/detections/label_{idx}",
                        rr.TextLog(label_text)
                    )
                    
                    rr.log(
                        f"world/detections/center_{idx}",
                        rr.Points3D(
                            [center_3d],
                            colors=[color],
                            radii=[0.01]
                        )
                    )
            
            except Exception as e:
                print(f"[경고] 탐지 {idx} 처리 실패: {e}")
                continue
    
    @staticmethod
    def _get_color_palette(num_classes: int) -> Dict[int, Tuple[int, int, int]]:
        """
        클래스별 색상 팔레트 생성
        
        Args:
            num_classes: 클래스 개수
        
        Returns:
            Dict[class_id] = (R, G, B) 색상
        """
        colors = [
            (0, 255, 0),      # 초록색
            (255, 0, 0),      # 빨강색
            (0, 0, 255),      # 파랑색
            (255, 255, 0),    # 노랑색
            (255, 0, 255),    # 마젠타
            (0, 255, 255),    # 시안
            (255, 128, 0),    # 주황색
            (128, 0, 255),    # 보라색
        ]
        
        color_map = {}
        for i in range(num_classes):
            color_map[i] = colors[i % len(colors)]
        
        return color_map
