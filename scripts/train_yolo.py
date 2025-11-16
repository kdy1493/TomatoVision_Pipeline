import os
import autorootcwd
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_yolo.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 1. 모드 설정 (YAML에서 'mode' 읽기, 없으면 'train'이 기본)
    mode = cfg.get('mode', 'train')

    # DETR와 동일한 방식: 설정값으로 GPU 강제 설정 (외부 지정보다 우선)
    # 튜닝 시에는 device 설정이 다르게 적용될 수 있으므로 train 모드에서만 강제
    if mode == 'train' and 'train' in cfg and 'device' in cfg['train']:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['train']['device'])

    # 튜닝 모드는 tune 설정의 'device'를 따름 (model.tune() 인자로 전달)
    model = YOLO(cfg['model'])

    # 2. 데이터 설정 (기존 로직 동일)
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

    # 3. 모드에 따른 분기
    if mode == 'train':
        print("--- [ Train Mode ] ---")
        if 'train' not in cfg:
            raise ValueError("YAML에 'train' 설정 블록이 없습니다.")
        
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
            # device는 os.environ으로 설정했거나, 명시 안하면 자동 선택됨
        )

    elif mode == 'tune':
        print("--- [ Tune Mode ] ---")
        if 'tune' not in cfg:
            raise ValueError("YAML에 'tune' 설정 블록이 없습니다.")
            
        tune_cfg = cfg['tune']
        
        # model.tune() 호출
        results = model.tune(
            data=data_yaml,
            # 튜닝 관련 핵심 파라미터
            iterations=tune_cfg['iterations'],
            epochs=tune_cfg['epochs'],  # 튜닝 1회당 에포크
            
            # 고정할 파라미터 (YAML에서 가져오기)
            imgsz=tune_cfg.get('imgsz', 640),
            batch=tune_cfg.get('batch', 16),
            optimizer=tune_cfg.get('optimizer', 'AdamW'),
            
            # 저장 경로
            project=tune_cfg.get('project', 'yolo_exp/'),
            name=tune_cfg.get('name', 'tune'),
            
            # 기타 튜닝 옵션
            device=tune_cfg.get('device', None),  # None이면 자동, 0 또는 '0,1' 등 지정 가능
            plots=tune_cfg.get('plots', True),
            save=tune_cfg.get('save', True),
            val=tune_cfg.get('val', True),
        )
        
        print(f"\n튜닝 완료. 최적 하이퍼파라미터:")
        if hasattr(results, 'best_hyperparameters'):
            print(f"  {results.best_hyperparameters}")
        if hasattr(results, 'save_dir'):
            print(f"결과는 {results.save_dir} 에 저장되었습니다.")
        else:
            print("튜닝 결과를 확인하세요.")

    else:
        raise ValueError(f"지원하지 않는 mode입니다: {mode} ('train' 또는 'tune'만 가능)")


if __name__ == '__main__':
    main()


