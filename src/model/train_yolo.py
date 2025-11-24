"""
YOLO 모델 학습 및 튜닝 모듈

사용법:
    # 직접 실행
    python -m src.model.train_yolo --config configs/train_yolo_exp1.yaml
    
    # 통합 스크립트 사용
    python scripts/yolo.py --config configs/yolo.yaml --mode train
    python scripts/yolo.py --config configs/yolo.yaml --mode tune
"""
import os
import autorootcwd
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO


def get_data_yaml(cfg):
    """YAML 설정에서 data.yaml 경로를 찾는 공통 함수"""
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
        raise FileNotFoundError('YOLO data.yaml 경로를 찾을 수 없습니다. YAML의 data.data_yaml 또는 data.dataset_root를 확인하세요.')
    return data_yaml


def train_mode(cfg):
    """YOLO 모델 학습"""
    print("--- [ Train Mode ] ---")
    
    if 'train' not in cfg:
        raise ValueError("YAML에 'train' 설정 블록이 없습니다.")
    
    # GPU 설정
    if 'device' in cfg['train']:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['train']['device'])
    
    # 모델 로드
    model = YOLO(cfg['model'])
    
    # 데이터 설정
    data_yaml = get_data_yaml(cfg)
    
    # 학습 설정
    train_cfg = cfg['train']
    model.train(
        data=data_yaml,
        epochs=train_cfg['epochs'],
        imgsz=train_cfg['imgsz'],
        batch=train_cfg['batch'],
        optimizer=train_cfg.get('optimizer', 'AdamW'),
        lr0=train_cfg.get('lr0', 1e-3),
        weight_decay=train_cfg.get('weight_decay', 1e-4),
        cos_lr=train_cfg.get('scheduler', 'Cosine') == 'Cosine',
        patience=train_cfg.get('patience', 50),
        mixup=train_cfg.get('mixup', 0.0),
        seed=train_cfg.get('seed', None),
        project=train_cfg.get('project', 'yolo_exp/'),
        name=train_cfg.get('name', 'detect'),
        workers=train_cfg.get('workers', 0),  # Windows multiprocessing 이슈 방지
    )


def tune_mode(cfg):
    """YOLO 모델 하이퍼파라미터 튜닝"""
    print("--- [ Tune Mode ] ---")
    
    if 'tune' not in cfg:
        raise ValueError("YAML에 'tune' 설정 블록이 없습니다.")
    
    # 모델 로드
    model = YOLO(cfg['model'])
    
    # 데이터 설정
    data_yaml = get_data_yaml(cfg)
    
    # 튜닝 설정
    tune_cfg = cfg['tune']
    results = model.tune(
        data=data_yaml,
        iterations=tune_cfg['iterations'],
        epochs=tune_cfg['epochs'],
        imgsz=tune_cfg.get('imgsz', 640),
        batch=tune_cfg.get('batch', 16),
        optimizer=tune_cfg.get('optimizer', 'AdamW'),
        project=tune_cfg.get('project', 'yolo_exp/'),
        name=tune_cfg.get('name', 'tune'),
        device=tune_cfg.get('device', None),
        plots=tune_cfg.get('plots', True),
        save=tune_cfg.get('save', True),
        val=tune_cfg.get('val', True),
    )
    
    print(f"\n튜닝 완료. 최적 하이퍼파라미터:")
    if hasattr(results, 'best_hyperparameters'):
        print(f"  {results.best_hyperparameters}")
    if hasattr(results, 'save_dir'):
        print(f"결과는 {results.save_dir} 에 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_yolo.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 모드 설정 (YAML에서 'mode' 읽기, 없으면 'train'이 기본)
    mode = cfg.get('mode', 'train')

    # 모드에 따른 분기
    if mode == 'train':
        train_mode(cfg)
    elif mode == 'tune':
        tune_mode(cfg)
    else:
        raise ValueError(f"지원하지 않는 mode입니다: {mode} ('train' 또는 'tune'만 가능)")


if __name__ == '__main__':
    main()


