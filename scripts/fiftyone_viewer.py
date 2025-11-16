"""
FiftyOne 데이터셋 뷰어 스크립트 (설정 파일 기반)
=====================================================

이 스크립트는 YAML 설정 파일에 정의된 정보를 바탕으로
COCO 또는 YOLO 형식의 객체 탐지 데이터셋을 FiftyOne으로 로드하여 시각화합니다.

[사용 방법]

1. `configs/fiftyone_viewer.yaml` 파일을 열어 시각화할 데이터셋 정보를 수정합니다.

2. 아래 명령어를 실행합니다.
   python scripts/fiftyone_viewer.py --config config/fiftyone_viewer.yaml

[설정 파일 인자 설명]
format: 데이터셋 형식 ('coco' 또는 'yolo').
image_path: 이미지 파일이 있는 디렉토리 경로.
labels_path: 라벨 파일 경로 (COCO의 경우 .json 파일, YOLO의 경우 .txt 파일들이 있는 디렉토리).
classes: (YOLO 전용) 클래스 이름이 정의된 .names 파일 경로.
dataset_name: FiftyOne에 표시될 데이터셋의 이름.
"""
import fiftyone as fo
import argparse
import os
import yaml
from types import SimpleNamespace

def load_dataset(config):
    """
    제공된 설정을 기반으로 FiftyOne 데이터셋을 로드합니다.
    GT와 예측 결과를 함께 로드할 수 있습니다.
    """
    
    # 데이터셋이 이미 존재하면 삭제하여 새로 로드합니다.
    if fo.dataset_exists(config.dataset_name):
        fo.delete_dataset(config.dataset_name)
        print(f"Deleted existing dataset: '{config.dataset_name}'")

    print(f"Loading dataset '{config.dataset_name}' from format '{config.format}'...")

    dataset_type = None
    kwargs = {}

    # -----------------------------------------------------------------
    # 포맷별 로드 설정
    # -----------------------------------------------------------------
    
    if config.format == 'coco':
        dataset_type = fo.types.COCODetectionDataset
        kwargs['data_path'] = config.image_path
        kwargs['labels_path'] = config.labels_path # COCO는 labels.json 파일 경로
        kwargs['label_field'] = 'ground_truth'  # GT는 'ground_truth' 필드에 저장
        
    elif config.format == 'yolo':
        dataset_type = fo.types.YOLOv5Dataset
        
        # dataset_yaml 파일 필수 (YOLOv5 표준 방식)
        if not hasattr(config, 'dataset_yaml') or not config.dataset_yaml:
            print("Error: YOLO format requires 'dataset_yaml' field in config.")
            print("Please provide the path to your YOLOv5 data.yaml file.")
            return None
        
        if not os.path.exists(config.dataset_yaml):
            print(f"Error: dataset_yaml file not found at '{config.dataset_yaml}'.")
            return None
        
        kwargs['yaml_path'] = config.dataset_yaml
        if hasattr(config, 'split') and config.split:
            kwargs['split'] = config.split  # 'train' 또는 'val'
        kwargs['label_field'] = getattr(config, 'label_field', 'ground_truth')
        print(f"Using dataset_yaml: '{config.dataset_yaml}'")
        if hasattr(config, 'split') and config.split:
            print(f"Loading split: '{config.split}'")

    else:
        print(f"Error: Unsupported format '{config.format}'.")
        print("Supported formats: 'coco', 'yolo'")
        return

    # -----------------------------------------------------------------
    # 데이터셋 로드
    # -----------------------------------------------------------------
    
    try:
        dataset = fo.Dataset.from_dir(
            dataset_type=dataset_type,
            name=config.dataset_name,
            **kwargs
        )
        
        # 데이터셋을 영구적으로 저장 (선택 사항)
        dataset.persistent = True

        print(f"\nSuccessfully loaded {len(dataset)} samples with ground truth.")
        
        # 예측 결과가 있으면 추가 로드
        if hasattr(config, 'predictions_path') and config.predictions_path:
            if os.path.exists(config.predictions_path):
                print(f"\nLoading predictions from: {config.predictions_path}")
                try:
                    import json
                    from collections import defaultdict
                    
                    # COCO 예측 JSON 로드
                    with open(config.predictions_path, 'r') as f:
                        pred_data = json.load(f)
                    
                    print(f"Predictions JSON contains:")
                    print(f"  - {len(pred_data.get('images', []))} images")
                    print(f"  - {len(pred_data.get('annotations', []))} annotations")
                    print(f"  - {len(pred_data.get('categories', []))} categories")
                    
                    # 카테고리 ID -> 이름 매핑
                    cat_id_to_name = {cat['id']: cat['name'] for cat in pred_data['categories']}
                    
                    # 이미지 ID -> 파일명 매핑
                    id_to_filename = {img['id']: img['file_name'] for img in pred_data['images']}
                    
                    # 파일명 -> 이미지 크기 매핑 (COCO JSON에서 가져옴)
                    filename_to_size = {}
                    for img in pred_data['images']:
                        filename_to_size[img['file_name']] = (img['width'], img['height'])
                    
                    # 이미지별 예측 결과 그룹화
                    img_to_anns = defaultdict(list)
                    for ann in pred_data['annotations']:
                        img_to_anns[ann['image_id']].append(ann)
                    
                    # 각 샘플에 예측 결과 추가
                    pred_count = 0
                    for sample in dataset:
                        # 파일명으로 이미지 ID 찾기
                        filename = os.path.basename(sample.filepath)
                        img_id = None
                        for iid, fname in id_to_filename.items():
                            if fname == filename:
                                img_id = iid
                                break
                        
                        if img_id is None:
                            continue
                        
                        # 해당 이미지의 예측 결과 가져오기
                        anns = img_to_anns.get(img_id, [])
                        if not anns:
                            sample["predictions"] = fo.Detections(detections=[])
                            sample.save()
                            continue
                        
                        # 이미지 크기 가져오기 (우선순위: COCO JSON > sample.metadata > 이미지 파일 직접 읽기)
                        img_w, img_h = None, None
                        if filename in filename_to_size:
                            img_w, img_h = filename_to_size[filename]
                        elif sample.metadata and sample.metadata.width and sample.metadata.height:
                            img_w = sample.metadata.width
                            img_h = sample.metadata.height
                        else:
                            # 이미지 파일에서 직접 읽기
                            try:
                                from PIL import Image
                                with Image.open(sample.filepath) as img:
                                    img_w, img_h = img.size
                            except Exception as e:
                                print(f"Warning: Could not get image size for {filename}: {e}")
                                continue
                        
                        if img_w is None or img_h is None:
                            print(f"Warning: Could not determine image size for {filename}, skipping predictions")
                            continue
                        
                        # FiftyOne Detections 생성
                        detections = []
                        for ann in anns:
                            # COCO bbox: [x, y, width, height] (absolute pixels)
                            x, y, w, h = ann['bbox']
                            
                            # FiftyOne은 [x, y, width, height] (normalized [0, 1])
                            bounding_box = [
                                x / img_w,
                                y / img_h,
                                w / img_w,
                                h / img_h
                            ]
                            
                            detection = fo.Detection(
                                label=cat_id_to_name.get(ann['category_id'], str(ann['category_id'])),
                                bounding_box=bounding_box,
                                confidence=ann.get('score', 1.0)
                            )
                            detections.append(detection)
                            pred_count += 1
                        
                        # 샘플에 예측 결과 추가
                        sample["predictions"] = fo.Detections(detections=detections)
                        sample.save()
                    
                    print(f"✓ Successfully loaded {pred_count} predictions into 'predictions' field.")
                    
                except Exception as pred_e:
                    print(f"Warning: Failed to load predictions: {pred_e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Warning: Predictions file not found: {config.predictions_path}")
        
        print(f"Dataset '{config.dataset_name}' is ready.")
        return dataset

    except Exception as e:
        print(f"\n--- 데이터 로드 중 오류 발생 ---")
        import traceback
        traceback.print_exc()
        #print(e)
        print("----------------------------------\n")
        print("경로와 형식이 올바른지 확인하세요.")
        if config.format == 'coco':
            print("> COCO 형식은 다음을 예상합니다:")
            print(f"  image_path: {config.image_path} (이미지 폴더)")
            print(f"  labels_path: {config.labels_path} (annotations.json 파일)")
            if hasattr(config, 'predictions_path'):
                print(f"  predictions_path: {config.predictions_path} (예측 결과 JSON, 선택사항)")
        elif config.format == 'yolo':
            print("> YOLO 형식은 다음을 예상합니다:")
            print(f"  dataset_yaml: (YOLOv5 data.yaml 파일 경로, 필수)")
            if hasattr(config, 'split'):
                print(f"  split: (선택사항: 'train' 또는 'val')")
            print(f"  label_field: (선택사항, 기본값: 'ground_truth')")
        return None

def main():
    parser = argparse.ArgumentParser(description="Load Object Detection datasets into FiftyOne Viewer using a config file.")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )

    args = parser.parse_args()

    # Load config from YAML file
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        # Convert dict to namespace object for dot notation access (e.g., config.format)
        config = SimpleNamespace(**config_dict)
    except FileNotFoundError:
        print(f"Error: Config file not found at '{args.config}'")
        return
    except Exception as e:
        print(f"Error parsing YAML file: {e}")
        return
    
    dataset = load_dataset(config)
    
    if dataset:
        # FiftyOne 앱 실행
        print("Launching FiftyOne App...")
        session = fo.launch_app(dataset)

        if session:
            print("FiftyOne App is running. The session will be kept alive.")
            print("Press Ctrl+C in this terminal to exit.")
            session.wait()
            print("Session ended.")
        else:
            print("Failed to launch FiftyOne App session.")

if __name__ == "__main__":
    main()