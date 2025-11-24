"""
통합 YOLO 스크립트 - 학습, 예측을 하나로

사용법:
    # 학습
    python scripts/yolo.py --config configs/yolo.yaml --mode train
    
    # 예측
    python scripts/yolo.py --config configs/yolo.yaml --mode predict

YAML 설정에서 mode를 지정하거나 --mode 인자로 지정할 수 있습니다.
"""
import autorootcwd
import argparse
import yaml

# 각 모듈에서 함수 import
from src.model.train_yolo import train_mode, tune_mode
from src.model.predict_yolo import predict_mode


def main():
    parser = argparse.ArgumentParser(description="통합 YOLO 스크립트 - 학습, 예측")
    parser.add_argument('--config', type=str, required=True, help='설정 파일 경로')
    parser.add_argument('--mode', type=str, choices=['train', 'tune', 'predict'], 
                       help='실행 모드 (YAML의 mode보다 우선)')
    args = parser.parse_args()
    
    # 설정 파일 로드
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 모드 결정 (인자 > YAML)
    mode = args.mode if args.mode else cfg.get('mode', 'train')
    
    if mode == 'train':
        train_mode(cfg)
    elif mode == 'tune':
        tune_mode(cfg)
    elif mode == 'predict':
        predict_mode(cfg)
    else:
        raise ValueError(f"지원하지 않는 mode입니다: {mode} ('train', 'tune', 'predict' 중 하나)")


if __name__ == '__main__':
    main()

