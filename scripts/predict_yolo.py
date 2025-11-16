"""
YOLO 모델을 사용하여 이미지에 대한 예측을 수행하고 결과를 저장하는 스크립트

사용법:
python scripts/predict_yolo.py --config configs/predict_yolo_exp1.yaml
"""
import os
import autorootcwd
import argparse
import yaml
import json
import re
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image

def predict_dataset(model_path, data_yaml, split='val', conf_thres=0.25):
    """
    데이터셋에 대한 예측을 수행하고 COCO 형식으로 결과를 저장합니다.
    """
    # 데이터 설정 로드
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # 이미지 경로 설정
    dataset_path = Path(data_config['path'])
    if split == 'val':
        split_path = data_config.get('val', 'val/images')
    else:
        split_path = data_config.get('train', 'train/images')
    
    img_dir = dataset_path / split_path
    if not img_dir.exists():
        # 'images'가 포함된 경우 제거하고 다시 시도
        if '/images' in split_path:
            split_path = split_path.replace('/images', '')
        img_dir = dataset_path / split_path
    
    print(f"Loading images from: {img_dir}")
    
    # 모델 로드
    model = YOLO(model_path)
    
    # COCO 형식의 결과를 위한 데이터 구조
    coco_output = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # 클래스 정보 추가
    names = data_config.get('names', [])
    for idx, name in enumerate(names):
        coco_output['categories'].append({
            'id': int(idx),
            'name': name,
            'supercategory': 'none'
        })
    
    # YOLO 모델로 디렉토리 전체 예측 수행 (간단한 API 사용)
    print(f"\nProcessing images in: {img_dir}")
    results = model(str(img_dir), conf=conf_thres, verbose=False)
    
    # 진행 상황 표시를 위한 tqdm 설정
    annotation_id = 0
    for img_id, result in enumerate(tqdm(results)):
        img_path = Path(result.path)
        img_file = img_path.name

        # ✅ 원본 크기: (H, W) 반환
        orig_h, orig_w = result.orig_shape  # tuple like (height, width)

        coco_output['images'].append({
            'id': img_id,
            'file_name': img_file,
            'width': int(orig_w),    # ✅ W
            'height': int(orig_h),   # ✅ H
        })

        if result.boxes is not None:
            # 선택: 바운딩 박스 좌표를 이미지 경계로 클리핑(안전)
            def clip(v, lo, hi): return max(lo, min(hi, float(v)))

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # pixel coords on original image
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # ✅ 경계 클리핑(옵션이지만 추천)
                x1 = clip(x1, 0, orig_w)
                x2 = clip(x2, 0, orig_w)
                y1 = clip(y1, 0, orig_h)
                y2 = clip(y2, 0, orig_h)

                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)

                coco_output['annotations'].append({
                    'id': annotation_id,
                    'image_id': img_id,
                    'category_id': cls,         # 필요 시 1부터 시작하도록 +1
                    'bbox': [x1, y1, bw, bh],   # COCO: [x, y, w, h]
                    'area': bw * bh,
                    'score': conf,
                    'iscrowd': 0
                })
                annotation_id += 1
    
    return coco_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='설정 파일 경로 (config.yaml)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 모델 경로
    model_path = cfg['model']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'모델 파일을 찾을 수 없습니다: {model_path}')

    # 데이터 설정: data.data_yaml 우선, 없으면 data.dataset_root/data.yaml 사용
    data_yaml = None
    if 'data' in cfg and isinstance(cfg['data'], dict):
        if 'data_yaml' in cfg['data'] and cfg['data']['data_yaml']:
            data_yaml = cfg['data']['data_yaml']
        elif 'dataset_root' in cfg['data'] and cfg['data']['dataset_root']:
            root = Path(cfg['data']['dataset_root'])
            candidate = root / 'data.yaml'
            if candidate.exists():
                data_yaml = str(candidate)
    if not data_yaml:
        raise FileNotFoundError('YOLO data.yaml 경로를 찾을 수 없습니다. config의 data.data_yaml 또는 data.dataset_root를 확인하세요.')

    # 예측 설정
    split = cfg['predict'].get('split', 'val')
    conf_thres = cfg['predict'].get('conf', 0.25)
    
    # 출력 파일 경로: 모델 경로에서 실험 디렉토리 추출
    model_path_obj = Path(model_path)
    # weights/best.pt 또는 weights/last.pt를 제거하여 실험 디렉토리 찾기
    if model_path_obj.parent.name == 'weights':
        exp_dir = model_path_obj.parent.parent  # yolo_exp/yolo12n_laboro_exp1/
    else:
        exp_dir = model_path_obj.parent  # 실험 디렉토리 직접 지정된 경우
    
    # 출력 파일명
    if 'output' in cfg['predict'] and cfg['predict']['output']:
        output_filename = cfg['predict']['output']
    else:
        # 자동 생성: prediction_val_exp1.json 형식
        exp_name = exp_dir.name  # yolo12n_laboro_exp1
        # exp1, exp2 등 추출 (exp 뒤의 숫자)
        exp_match = re.search(r'exp(\d+)', exp_name)
        if exp_match:
            exp_num = exp_match.group(1)
            output_filename = f"prediction_{split}_exp{exp_num}.json"
        else:
            # exp 패턴이 없으면 전체 이름 사용
            output_filename = f"prediction_{split}_{exp_name}.json"
    
    output_path = exp_dir / output_filename
    os.makedirs(exp_dir, exist_ok=True)

    print(f"Using model: {model_path}")
    print(f"Dataset config: {data_yaml}")
    print(f"Split: {split}")
    print(f"Confidence threshold: {conf_thres}")
    print(f"Output will be saved to: {output_path}")
    
    # 예측 수행
    results = predict_dataset(model_path, data_yaml, split, conf_thres)
    
    # 결과 저장
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nPrediction completed!")
    print(f"Total images: {len(results['images'])}")
    print(f"Total predictions: {len(results['annotations'])}")
    print(f"Results saved to: {output_path}")

if __name__ == '__main__':
    main()