"""
비디오 입력 → YOLO → FastSAM → 이진 마스크 영상 출력 파이프라인 (patched)

사용법:
python scripts/video_mask_pipeline.py --config configs/video_mask_pipeline.yaml
"""
import os
import autorootcwd
import argparse
import yaml
import cv2
import numpy as np
import torch
import time
import subprocess
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics import FastSAM


def process_video(
    video_path: str,
    yolo_model_path: str,
    fastsam_model_path: str,
    output_path: str,
    yolo_output_path: str = None,
    conf_thres: float = 0.25,
    device: str = None,
    fps: int = None,
    tomato_class_ids: list | None = None,   # [NEW] 클래스 ID로 필터
    tomato_class_names: list | None = None, # [NEW] 이름으로 필터(백업)
    enhance_input: bool = False,            # [NEW] 밝기/대비 보정 옵션
    imgsz: int = 640,                       # [NEW] YOLO 입력 크기 조정
    iou_thres: float | None = None,         # [NEW] NMS IOU (버전 호환 시)
):
    """
    비디오를 처리하여 이진 마스크 영상과 YOLO 결과 영상을 생성합니다.
    """
    # 디바이스 설정
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    # 모델 로드
    print(f"Loading YOLO model from: {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)
    # [FIX] YOLO도 디바이스 일치
    try:
        yolo_model.to(device)
    except Exception:
        pass

    print(f"Loading FastSAM model from: {fastsam_model_path}")
    fastsam_model = FastSAM(fastsam_model_path)

    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")

    # 비디오 정보
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps is None:
        fps = original_fps

    print(f"비디오 정보: {width}x{height}, {total_frames} frames, {original_fps} FPS")

    # 출력 경로 처리 (확장자 대소문자 대응) [FIX]
    output_path_obj = Path(output_path)
    input_video_name = Path(video_path).stem

    if not output_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # [FIX]
        output_dir = output_path_obj
        output_path = str(output_dir / f"{input_video_name}_masked.mp4")
        print(f"출력 경로를 폴더로 인식: {output_dir}")
        print(f"출력 파일명 생성: {output_path}")
    else:
        output_dir = output_path_obj.parent

    # YOLO 결과 영상 경로
    if yolo_output_path is None:
        yolo_output_path = str(Path(output_path).parent / f"{Path(output_path).stem}_yolo{Path(output_path).suffix}")
    elif not yolo_output_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # [FIX]
        yolo_output_dir = Path(yolo_output_path)
        yolo_output_path = str(yolo_output_dir / f"{input_video_name}_masked_yolo.mp4")

    # 출력 디렉토리 생성
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(yolo_output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"이진 마스크 출력: {output_path}")
    print(f"YOLO 결과 출력: {yolo_output_path}")

    # ffmpeg 파이프라인 준비
    print("H.264 코덱 사용 (ffmpeg)")

    # [FIX] -loglevel error 로 파이프 잠금 위험 완화
    ffmpeg_cmd_mask = [
        'ffmpeg', '-loglevel', 'error',
        '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'gray',
        '-r', str(fps), '-i', '-',
        '-an', '-vcodec', 'libx264', '-preset', 'medium',
        '-crf', '23', '-pix_fmt', 'yuv420p', output_path
    ]

    ffmpeg_cmd_yolo = [
        'ffmpeg', '-loglevel', 'error',
        '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-',
        '-an', '-vcodec', 'libx264', '-preset', 'medium',
        '-crf', '23', '-pix_fmt', 'yuv420p', yolo_output_path
    ]

    try:
        proc_mask = subprocess.Popen(ffmpeg_cmd_mask, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        proc_yolo = subprocess.Popen(ffmpeg_cmd_yolo, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise ValueError("ffmpeg가 설치되어 있지 않습니다. ffmpeg를 설치해주세요.")
    except Exception as e:
        raise ValueError(f"ffmpeg 프로세스를 시작할 수 없습니다: {e}")

    frame_count = 0
    start_time = time.time()
    frame_times = []
    last_print_time = time.time()
    print_interval = 1.0

    # 필터 기준(완화 가능)
    MAX_AREA_RATIO = 0.8
    MIN_ASPECT, MAX_ASPECT = 0.1, 10.0

    # names 캐시
    names = getattr(yolo_model, "names", {})
    names_is_dict = isinstance(names, dict)

    try:
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                ret, frame_bgr = cap.read()  # OpenCV는 BGR
                if not ret:
                    break

                frame_start_time = time.time()

                # FastSAM용 RGB [유지]
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # (선택) 입력 강화 [NEW]
                yolo_input_bgr = frame_bgr
                if enhance_input:
                    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
                    h, s, v = cv2.split(hsv)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    v = clahe.apply(v)
                    hsv = cv2.merge([h, s, v])
                    yolo_input_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                # 1) YOLO로 객체 감지 — [FIX] YOLO에는 BGR 입력
                yolo_start = time.time()
                predict_kwargs = dict(conf=conf_thres, imgsz=imgsz, device=device, verbose=False)
                if iou_thres is not None:
                    predict_kwargs["iou"] = iou_thres
                if tomato_class_ids is not None:
                    predict_kwargs["classes"] = tomato_class_ids

                yolo_results = yolo_model(yolo_input_bgr, **predict_kwargs)
                yolo_result = yolo_results[0]
                yolo_time = time.time() - yolo_start

                # 2) 박스 필터링
                tomato_boxes = []
                tomato_indices = []

                if yolo_result.boxes is not None and len(yolo_result.boxes) > 0:
                    img_height, img_width = yolo_input_bgr.shape[:2]

                    for idx, box in enumerate(yolo_result.boxes):
                        cls_id = int(box.cls[0]) if hasattr(box, "cls") else -1

                        # 클래스 필터: ID 우선, 없으면 이름 필터
                        passed = False
                        if tomato_class_ids is not None:
                            if cls_id in set(tomato_class_ids):
                                passed = True
                        elif tomato_class_names is not None:
                            if names_is_dict:
                                cls_name = names.get(cls_id, str(cls_id))
                            else:
                                # 리스트/기타
                                try:
                                    cls_name = names[cls_id]
                                except Exception:
                                    cls_name = str(cls_id)
                            if cls_name in set(tomato_class_names):
                                passed = True
                        else:
                            # 필터 미지정 시 전부 통과
                            passed = True

                        if not passed:
                            continue

                        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()
                        box_width = max(1.0, x2 - x1)
                        box_height = max(1.0, y2 - y1)
                        box_area = box_width * box_height
                        box_ratio = box_width / box_height
                        area_ratio = box_area / (img_width * img_height)

                        # [완화된 필터] — 초기 디버깅 시엔 꺼도 좋음
                        if (area_ratio < MAX_AREA_RATIO) and (MIN_ASPECT < box_ratio < MAX_ASPECT):
                            tomato_boxes.append(box.xyxy[0])
                            tomato_indices.append(idx)
                        else:
                            # 필요시 디버그 출력
                            pass

                    yolo_box_coords = torch.stack(tomato_boxes) if len(tomato_boxes) > 0 else None
                else:
                    yolo_box_coords = None

                # 3) YOLO 시각화 — plot()은 BGR 반환. [FIX] 그대로 사용
                if yolo_result.boxes is not None and len(tomato_indices) > 0:
                    try:
                        yolo_annotated_bgr = yolo_result.plot(boxes=yolo_result.boxes[tomato_indices])
                    except Exception:
                        # 버전 호환 이슈 시 수동 드로잉
                        yolo_annotated_bgr = yolo_input_bgr.copy()
                        for i in tomato_indices:
                            b = yolo_result.boxes[i]
                            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                            cv2.rectangle(yolo_annotated_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cls_id = int(b.cls[0]) if hasattr(b, "cls") else -1
                            if names_is_dict:
                                cls_name = names.get(cls_id, str(cls_id))
                            else:
                                try:
                                    cls_name = names[cls_id]
                                except Exception:
                                    cls_name = str(cls_id)
                            conf = float(b.conf[0]) if hasattr(b, "conf") else 1.0
                            cv2.putText(yolo_annotated_bgr, f"{cls_name} {conf:.2f}", (x1, max(0, y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    yolo_annotated_bgr = yolo_input_bgr.copy()

                # 리사이즈 일치
                if yolo_annotated_bgr.shape[:2] != (height, width):
                    yolo_annotated_bgr = cv2.resize(yolo_annotated_bgr, (width, height), interpolation=cv2.INTER_LINEAR)

                # 4) FastSAM — 토마토 감지된 경우에만
                fastsam_time = 0.0
                binary_image = np.zeros((height, width), dtype=np.uint8)
                if yolo_box_coords is not None and len(yolo_box_coords) > 0:
                    fastsam_start = time.time()
                    fastsam_results = fastsam_model.predict(
                        source=frame_rgb,           # FastSAM은 RGB
                        bboxes=yolo_box_coords,
                        device=device,
                        retina_masks=True,
                        imgsz=imgsz,
                        conf=0.5,
                        iou=0.9,
                        verbose=False
                    )
                    fastsam_time = time.time() - fastsam_start

                    # [FIX] masks 안전 처리
                    if len(fastsam_results) > 0 and getattr(fastsam_results[0], "masks", None) is not None:
                        masks = fastsam_results[0].masks.data
                        if torch.is_tensor(masks):
                            masks = masks.detach().cpu().numpy()
                        else:
                            masks = np.asarray(masks)
                        if masks.ndim >= 2:
                            binary_mask = np.any(masks > 0, axis=0)
                        else:
                            binary_mask = masks > 0
                        binary_image = (binary_mask.astype(np.uint8) * 255)

                    if binary_image.shape != (height, width):
                        binary_image = cv2.resize(binary_image, (width, height), interpolation=cv2.INTER_NEAREST)

                # ffmpeg로 전송
                try:
                    proc_mask.stdin.write(binary_image.tobytes())
                    proc_yolo.stdin.write(yolo_annotated_bgr.tobytes())
                except BrokenPipeError:
                    try:
                        if proc_mask and proc_mask.stderr:
                            proc_mask.stderr.close()
                        if proc_yolo and proc_yolo.stderr:
                            proc_yolo.stderr.close()
                    except Exception:
                        pass
                    raise RuntimeError("ffmpeg 프로세스가 예기치 않게 종료되었습니다.")

                # 속도 로깅
                frame_time = time.time() - frame_start_time
                frame_times.append(frame_time)
                frame_count += 1

                current_time = time.time()
                if current_time - last_print_time >= print_interval:
                    avg_fps = frame_count / (current_time - start_time)
                    if len(frame_times) >= 30:
                        recent_sum = sum(frame_times[-30:])
                        recent_fps = 30.0 / recent_sum if recent_sum > 0 else 0
                        avg_frame_time = float(np.mean(frame_times[-30:]))
                    else:
                        recent_fps = avg_fps
                        avg_frame_time = float(np.mean(frame_times)) if frame_times else frame_time

                    pbar.set_postfix({
                        'FPS': f'{avg_fps:.2f}',
                        'Recent FPS': f'{recent_fps:.2f}',
                        'Frame Time': f'{avg_frame_time*1000:.1f}ms',
                        'YOLO': f'{yolo_time*1000:.1f}ms',
                        'FastSAM': f'{fastsam_time*1000:.1f}ms' if fastsam_time > 0 else '0ms'
                    })
                    last_print_time = current_time

                # (초기 50프레임 디버그) — 탐지 클래스/스코어 확인
                if frame_count <= 50 and yolo_result.boxes is not None:
                    for b in yolo_result.boxes:
                        c = int(b.cls[0]) if hasattr(b, "cls") else -1
                        conf = float(b.conf[0]) if hasattr(b, "conf") else -1.0
                        if names_is_dict:
                            cname = names.get(c, str(c))
                        else:
                            try:
                                cname = names[c]
                            except Exception:
                                cname = str(c)
                        # 필요 시 주석 해제
                        # print(f"[dbg] cls_id={c}, name={cname}, conf={conf:.2f}, xyxy={b.xyxy[0].tolist()}")

                pbar.update(1)

    finally:
        cap.release()
        # ffmpeg 종료
        try:
            proc_mask.stdin.close()
            proc_yolo.stdin.close()
            return_code_mask = proc_mask.wait()
            return_code_yolo = proc_yolo.wait()

            if return_code_mask != 0 and proc_mask.stderr:
                stderr_mask = proc_mask.stderr.read().decode('utf-8', errors='ignore')
                print(f"Warning: 이진 마스크 ffmpeg 오류 종료 (코드: {return_code_mask})")
                if stderr_mask:
                    print(f"ffmpeg stderr: {stderr_mask[-500:]}")

            if return_code_yolo != 0 and proc_yolo.stderr:
                stderr_yolo = proc_yolo.stderr.read().decode('utf-8', errors='ignore')
                print(f"Warning: YOLO 결과 ffmpeg 오류 종료 (코드: {return_code_yolo})")
                if stderr_yolo:
                    print(f"ffmpeg stderr: {stderr_yolo[-500:]}")
        except Exception as e:
            print(f"Warning: ffmpeg 프로세스 종료 중 오류: {e}")
            try:
                proc_mask.terminate()
                proc_yolo.terminate()
            except Exception:
                pass

    # 출력 파일 확인
    output_file = Path(output_path)
    if output_file.exists():
        file_size = output_file.stat().st_size / (1024 * 1024)
        print(f"\n이진 마스크 파일 생성 확인: {output_path} ({file_size:.2f} MB)")
    else:
        print(f"\nWarning: 이진 마스크 파일이 생성되지 않았습니다: {output_path}")

    yolo_output_file = Path(yolo_output_path)
    if yolo_output_file.exists():
        yolo_file_size = yolo_output_file.stat().st_size / (1024 * 1024)
        print(f"YOLO 결과 파일 생성 확인: {yolo_output_path} ({yolo_file_size:.2f} MB)")
    else:
        print(f"Warning: YOLO 결과 파일이 생성되지 않았습니다: {yolo_output_path}")

    # 통계
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_frame_time = float(np.mean(frame_times)) if frame_times else 0
    min_frame_time = float(np.min(frame_times)) if frame_times else 0
    max_frame_time = float(np.max(frame_times)) if frame_times else 0

    print(f"\n{'='*60}")
    print(f"처리 완료 통계")
    print(f"{'='*60}")
    print(f"총 프레임 수: {frame_count}")
    print(f"총 처리 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
    print(f"평균 처리 속도: {avg_fps:.2f} FPS")
    print(f"평균 프레임 처리 시간: {avg_frame_time*1000:.2f}ms")
    print(f"최소 프레임 처리 시간: {min_frame_time*1000:.2f}ms")
    print(f"최대 프레임 처리 시간: {max_frame_time*1000:.2f}ms")
    print(f"이진 마스크 파일: {output_path}")
    if output_file.exists():
        print(f"이진 마스크 파일 크기: {file_size:.2f} MB")
    print(f"YOLO 결과 파일: {yolo_output_path}")
    if yolo_output_file.exists():
        print(f"YOLO 결과 파일 크기: {yolo_file_size:.2f} MB")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="비디오 → YOLO → FastSAM → 이진 마스크 영상 파이프라인 (patched)")
    parser.add_argument('--config', type=str, required=True, help='설정 파일 경로 (YAML)')
    args = parser.parse_args()

    # 설정 파일 로드 (UTF-8 인코딩 명시)
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 필수 설정
    video_path = cfg['input']['video_path']
    yolo_model_path = cfg['models']['yolo']
    fastsam_model_path = cfg['models']['fastsam']
    output_path = cfg['output']['video_path']

    # 선택적 설정
    yolo_output_path = cfg.get('output', {}).get('yolo_video_path', None)
    conf_thres = cfg.get('predict', {}).get('conf', 0.25)
    device = cfg.get('predict', {}).get('device', None)
    fps = cfg.get('output', {}).get('fps', None)
    imgsz = cfg.get('predict', {}).get('imgsz', 640)
    iou_thres = cfg.get('predict', {}).get('iou', None)
    enhance_input = cfg.get('predict', {}).get('enhance_input', False)

    # 클래스 필터 (ID/이름 둘 중 하나 선택 가능)
    tomato_class_ids = cfg.get('predict', {}).get('tomato_class_ids', None)
    tomato_class_names = cfg.get('predict', {}).get('tomato_class_names', None)

    process_video(
        video_path=video_path,
        yolo_model_path=yolo_model_path,
        fastsam_model_path=fastsam_model_path,
        output_path=output_path,
        yolo_output_path=yolo_output_path,
        conf_thres=conf_thres,
        device=device,
        fps=fps,
        tomato_class_ids=tomato_class_ids,
        tomato_class_names=tomato_class_names,
        enhance_input=enhance_input,
        imgsz=imgsz,
        iou_thres=iou_thres,
    )


if __name__ == '__main__':
    main()
