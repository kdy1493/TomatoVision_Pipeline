"""
3D 바운딩박스 시각화 모듈 - YOLO 2D BBox Back-projection

개념 요약
---------
1. YOLO 2D bbox (u_min, v_min, u_max, v_max)
2. bbox 내부 depth의 중앙값(median depth) 계산
3. 4개 코너 픽셀을 같은 depth로 back-projection → front plane 4점
4. 깊이 방향(thickness)을 약간 뒤쪽으로 확장 → back plane 4점
5. 총 8개의 3D 코너로 직육면체(box) 구성
6. 포인트클라우드에 사용한 것과 **동일한 카메라→월드 변환행렬(T_cam_to_world)** 적용
7. Rerun으로 라인 스트립과 중심점을 log
"""

from typing import Tuple, Dict, Optional, List

import numpy as np
import rerun as rr


class BBox3DVisualizer:
    """
    YOLO 2D bbox를 Back-projection으로 3D 직육면체 박스로 변환하여 Rerun에 시각화하는 모듈.

    ⚠️ 좌표계 설계
    -------------
    - 이 클래스는 먼저 **카메라 좌표계(camera frame)**에서 3D 코너를 계산한다.
    - 그 다음, 포인트클라우드에 사용한 것과 완전히 동일한 4x4 변환행렬
      `T_cam_to_world` (camera → world/Rerun)를 적용한다.
    - 즉, point_cloud.py에서 사용한 transform을 그대로 전달해야
      포인트클라우드와 bbox가 같은 공간에 뜬다.
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depth_min: float = 0.4,
        depth_max: float = 1.5,
        thickness_ratio: float = 0.15,
        T_cam_to_world: Optional[np.ndarray] = None,
    ):
        """
        Args:
            fx, fy: 카메라 focal length (pixel 단위)
            cx, cy: principal point (pixel 단위)
            depth_min: depth 유효 범위 최소값 (m)
            depth_max: depth 유효 범위 최대값 (m)
            thickness_ratio: 박스 두께 비율 (mean_depth * thickness_ratio)
                             ex) 0.15면 깊이의 15%만큼 뒤쪽으로 box 확장
            T_cam_to_world: (4,4) 카메라→월드 변환 행렬 (포인트클라우드에 쓰는 것과 동일하게)
                            None이면 카메라 좌표계 == 월드 좌표계로 간주
        """
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.thickness_ratio = float(thickness_ratio)

        if T_cam_to_world is not None:
            T_cam_to_world = np.asarray(T_cam_to_world, dtype=np.float32)
            assert T_cam_to_world.shape == (4, 4), \
                "T_cam_to_world must be a 4x4 matrix"
        self.T_cam_to_world = T_cam_to_world

    # ------------------------------------------------------------------
    # 기본 유틸
    # ------------------------------------------------------------------
    def _back_project_pixel_cam(
        self, u_px: float, v_px: float, depth_m: float
    ) -> np.ndarray:
        """
        픽셀 좌표 → 카메라 좌표계 3D 점 (X_cam, Y_cam, Z_cam)

        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        z_cam = depth
        """
        x_cam = (u_px - self.cx) * depth_m / self.fx
        y_cam = (v_px - self.cy) * depth_m / self.fy
        z_cam = depth_m
        return np.array([x_cam, y_cam, z_cam], dtype=np.float32)

    def _to_world(self, pts_cam: np.ndarray) -> np.ndarray:
        """
        카메라 좌표계 점들을 월드 좌표계로 변환.

        Args:
            pts_cam: (N,3) camera frame 좌표

        Returns:
            (N,3) world/Rerun frame 좌표
        """
        if self.T_cam_to_world is None:
            # 변환행렬이 없으면 카메라 좌표계 == 월드 좌표계로 사용
            return pts_cam

        pts_cam = np.asarray(pts_cam, dtype=np.float32)
        N = pts_cam.shape[0]
        pts_h = np.concatenate(
            [pts_cam, np.ones((N, 1), dtype=np.float32)], axis=1
        )  # (N,4)

        pts_world_h = pts_h @ self.T_cam_to_world.T  # (N,4)
        pts_world = pts_world_h[:, :3] / pts_world_h[:, 3:4]
        return pts_world

    # ------------------------------------------------------------------
    # 3D 박스 생성
    # ------------------------------------------------------------------
    def _compute_box_corners_cam(
        self,
        bbox_2d: Tuple[float, float, float, float],
        depth_image: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        2D bbox + depth 중앙값으로 카메라 좌표계에서 3D 직육면체 corner 생성.

        Args:
            bbox_2d: (u_min, v_min, u_max, v_max) in pixel
            depth_image: (H, W) depth map [m]

        Returns:
            corners_cam: (8,3) 카메라 좌표계 8개 코너
            median_depth: bbox 내부 depth 중앙값
        """
        h, w = depth_image.shape[:2]
        u_min, v_min, u_max, v_max = map(int, bbox_2d)

        # 이미지 경계 클램프
        u_min = max(0, min(u_min, w - 1))
        u_max = max(0, min(u_max, w - 1))
        v_min = max(0, min(v_min, h - 1))
        v_max = max(0, min(v_max, h - 1))

        if u_min >= u_max:
            u_min, u_max = u_max, u_min
        if v_min >= v_max:
            v_min, v_max = v_max, v_min

        # 1) ROI depth 추출 & 유효값(median) 계산
        roi_depth = depth_image[v_min : v_max + 1, u_min : u_max + 1]
        valid = (roi_depth > self.depth_min) & (roi_depth < self.depth_max)
        valid_depths = roi_depth[valid]

        if valid_depths.size == 0:
            # 유효 depth가 전혀 없으면 중간값으로 fallback
            median_depth = 0.5 * (self.depth_min + self.depth_max)
        else:
            median_depth = float(np.median(valid_depths))

        # 2) bbox 4개 코너 픽셀 좌표
        corners_2d = np.array(
            [
                [u_min, v_min],  # 좌상
                [u_max, v_min],  # 우상
                [u_min, v_max],  # 좌하
                [u_max, v_max],  # 우하
            ],
            dtype=np.float32,
        )

        # front plane: median_depth
        front_pts_cam = np.stack(
            [
                self._back_project_pixel_cam(u, v, median_depth)
                for (u, v) in corners_2d
            ],
            axis=0,
        )  # (4,3)

        # 박스 두께 설정 (깊이 방향으로 뒤로 밀기)
        depth_thickness = max(0.05, median_depth * self.thickness_ratio)
        back_depth = median_depth + depth_thickness

        back_pts_cam = np.stack(
            [
                self._back_project_pixel_cam(u, v, back_depth)
                for (u, v) in corners_2d
            ],
            axis=0,
        )  # (4,3)

        corners_cam = np.concatenate([front_pts_cam, back_pts_cam], axis=0)  # (8,3)
        return corners_cam, median_depth

    # ------------------------------------------------------------------
    # Rerun 로깅
    # ------------------------------------------------------------------
    @staticmethod
    def _make_box_edges(corners: np.ndarray) -> List[List[List[float]]]:
        """
        8개 코너로부터 12개 엣지를 LineStrips3D용 세그먼트로 변환.
        corners: (8,3)
        """
        # corner index:
        # 0,1,2,3 : front, 4,5,6,7 : back
        edges = [
            (0, 1),
            (1, 3),
            (3, 2),
            (2, 0),  # front face
            (4, 5),
            (5, 7),
            (7, 6),
            (6, 4),  # back face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # connections
        ]

        segs: List[List[List[float]]] = []
        for i, j in edges:
            segs.append([corners[i].tolist(), corners[j].tolist()])
        return segs

    @staticmethod
    def _make_color_palette(num_classes: int) -> Dict[int, Tuple[int, int, int]]:
        """
        간단한 클래스별 RGB 팔레트.
        """
        base_colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (128, 0, 255),
        ]
        palette: Dict[int, Tuple[int, int, int]] = {}
        for cid in range(num_classes):
            palette[cid] = base_colors[cid % len(base_colors)]
        return palette

    # ------------------------------------------------------------------
    # 메인 API
    # ------------------------------------------------------------------
    def log_3d_bboxes_from_yolo(
        self,
        yolo_results,
        depth_image: np.ndarray,
        class_names: Optional[Dict[int, str]] = None,
        conf_thresh: float = 0.5,
        base_path: str = "world/detections",
        show_centers: bool = True,
    ):
        """
        Ultralytics YOLO `results[0]` 객체와 depth map을 받아
        3D 바운딩 박스를 Rerun에 그리는 함수.

        Args:
            yolo_results: Ultralytics YOLO results[0]
            depth_image: (H, W) depth map [m]
            class_names: {class_id: class_name}
            conf_thresh: confidence 필터링 임계값
            base_path: Rerun 엔티티 prefix
            show_centers: 박스 중앙에 포인트 찍을지 여부
        """
        if yolo_results.boxes is None or len(yolo_results.boxes) == 0:
            return

        depth_image = np.asarray(depth_image, dtype=np.float32)
        class_names = class_names or {}

        # 팔레트는 "알려진 클래스 수" 기반으로 만들되,
        # class_names가 비어 있으면 기본 8색 팔레트 생성
        num_classes_for_palette = max(len(class_names), 1)
        palette = self._make_color_palette(num_classes_for_palette)

        det_idx = 0
        for box in yolo_results.boxes:
            conf = float(box.conf[0]) if hasattr(box, "conf") else 1.0
            if conf < conf_thresh:
                continue

            cls_id = int(box.cls[0]) if hasattr(box, "cls") else 0
            cls_name = class_names.get(cls_id, f"class_{cls_id}")

            # xyxy: [u_min, v_min, u_max, v_max]
            bbox_xyxy = box.xyxy[0].detach().cpu().numpy()
            try:
                # 1) 카메라 좌표계에서 8개 코너 계산
                corners_cam, median_depth = self._compute_box_corners_cam(
                    bbox_xyxy, depth_image
                )

                # 2) 월드(또는 Rerun) 좌표계로 변환
                corners_world = self._to_world(corners_cam)

                # 3) 엣지 세그먼트 생성
                segs = self._make_box_edges(corners_world)
                color = palette.get(cls_id, (0, 255, 0))

                # 4) LineStrips3D로 log
                rr.log(
                    f"{base_path}/bbox3d_{det_idx}",
                    rr.LineStrips3D(
                        segs,
                        colors=[color] * len(segs),
                        radii=[0.003] * len(segs),
                    ),
                )

                # 5) 중앙 포인트 + 라벨 (옵션)
                if show_centers:
                    center_world = corners_world.mean(axis=0)
                    rr.log(
                        f"{base_path}/center_{det_idx}",
                        rr.Points3D(
                            [center_world],
                            radii=[0.02],
                            colors=[color],
                            labels=[f"{cls_name} ({median_depth:.2f}m)"],
                        ),
                    )

                det_idx += 1

            except Exception as e:
                print(f"[BBox3DVisualizer] bbox {det_idx} 처리 중 오류: {e}")
                continue


class DummySegmenter:
    """세그멘테이션 비활성화 시 사용되는 더미 세그멘터: 전체 이미지를 전경으로 간주."""

    def get_mask(self, image_np: np.ndarray) -> np.ndarray:
        return np.ones((image_np.shape[0], image_np.shape[1]), dtype=bool)
