''' 
laboro-tomato 데이터셋 다운로드 및 설정
kagglehub에서 데이터셋 다운로드 후 data/tomato_data/laboro-tomato 에 이동
test.json을 val.json으로 이름 변경
클래스 ID 재매핑 (6개 클래스 → 3개 클래스)
data.yaml 파일 생성
기존 kagglehub 캐시 디렉토리 삭제
3종 클래스로 변경 : fully_ripened (0), half_ripened (1), green (2)

실행 명령어: python scripts/download_dataset.py
'''
import autorootcwd
import kagglehub
import os
import shutil
import json
import yaml
from pathlib import Path


def download_dataset(dataset_name: str) -> str:
    """kagglehub에서 데이터셋을 다운로드합니다.
    
    Args:
        dataset_name: 다운로드할 데이터셋 이름 (예: "nexuswho/laboro-tomato")
    
    Returns:
        다운로드된 데이터셋의 경로
    """
    path = kagglehub.dataset_download(dataset_name)
    print(f"Path to dataset files: {path}")
    return path


def move_dataset(source_path: str, target_base_dir: str, dataset_subdir: str) -> str:
    """다운로드된 데이터셋을 타겟 디렉토리로 이동합니다.
    
    Args:
        source_path: 다운로드된 데이터셋의 원본 경로
        target_base_dir: 타겟 기본 디렉토리 (예: "data/tomato_data")
        dataset_subdir: 데이터셋 서브디렉토리 이름 (예: "laboro-tomato")
    
    Returns:
        이동된 데이터셋의 최종 경로
    """
    os.makedirs(target_base_dir, exist_ok=True)
    target_dir = os.path.join(target_base_dir, dataset_subdir)
    
    if os.path.exists(source_path):
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        
        shutil.move(source_path, target_dir)
        print(f"데이터셋이 {target_dir}로 이동되었습니다.")
        return target_dir
    else:
        print(f"경고: 다운로드 경로를 찾을 수 없습니다: {source_path}")
        return None


def rename_test_to_val(dataset_dir: str) -> bool:
    """test.json을 val.json으로 이름 변경 (val/images와 같은 이미지인 경우).
    
    Args:
        dataset_dir: 데이터셋 디렉토리 경로
    
    Returns:
        이름 변경 성공 여부
    """
    annotations_dir = Path(dataset_dir) / "annotations"
    test_json_path = annotations_dir / "test.json"
    val_images_dir = Path(dataset_dir) / "val" / "images"
    
    if not test_json_path.exists() or not val_images_dir.exists():
        if test_json_path.exists():
            print(f"경고: val/images 폴더를 찾을 수 없습니다. test.json 이름 변경을 건너뜁니다.")
        elif val_images_dir.exists():
            print(f"경고: test.json 파일을 찾을 수 없습니다.")
        return False
    
    # test.json에서 이미지 파일명 추출
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    
    test_images = set()
    if 'images' in test_data:
        for img in test_data['images']:
            if 'file_name' in img:
                test_images.add(img['file_name'])
            elif 'filename' in img:
                test_images.add(img['filename'])
    
    # val/images 폴더의 이미지 파일명 추출
    val_images = set()
    for img_file in val_images_dir.iterdir():
        if img_file.is_file():
            val_images.add(img_file.name)
    
    # 두 목록이 같으면 test.json을 val.json으로 이름 변경
    if test_images == val_images:
        val_json_path = annotations_dir / "val.json"
        shutil.move(str(test_json_path), str(val_json_path))
        print(f"test.json을 val.json으로 이름 변경 완료 (이미지 {len(test_images)}개 일치)")
        return True
    else:
        print(f"경고: test.json({len(test_images)}개)과 val/images({len(val_images)}개)의 이미지가 일치하지 않습니다.")
        return False


def remap_class_ids(dataset_dir: str) -> bool:
    """COCO annotation JSON과 YOLO 라벨 파일의 클래스 ID를 재매핑합니다.
    
    매핑 규칙:
    - fully_ripened → 0
    - half_ripened → 1
    - green → 2
    
    Args:
        dataset_dir: 데이터셋 디렉토리 경로
    
    Returns:
        재매핑 성공 여부
    """
    annotations_dir = Path(dataset_dir) / "annotations"
    
    # COCO annotation에서 클래스 매핑 생성
    coco_data = None
    if (annotations_dir / "train.json").exists():
        with open(annotations_dir / "train.json", 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
    elif (annotations_dir / "val.json").exists():
        with open(annotations_dir / "val.json", 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
    
    if not coco_data or 'categories' not in coco_data:
        print("경고: COCO annotation 파일을 찾을 수 없습니다. 클래스 ID 재매핑을 건너뜁니다.")
        return False
    
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    
    # 클래스 이름에서 접두사 제거하고 목표 YOLO ID 매핑 생성
    # fully_ripened → 0, half_ripened → 1, green → 2
    coco_id_to_yolo_id = {}
    class_name_to_yolo_id = {
        'fully_ripened': 0,
        'half_ripened': 1,
        'green': 2
    }
    
    for cat in categories:
        class_name = cat['name']
        # 접두사 제거 (b_ 또는 l_)
        if class_name.startswith('b_') or class_name.startswith('l_'):
            class_name = class_name[2:]
        
        # 목표 클래스에 매핑
        if class_name in class_name_to_yolo_id:
            coco_id_to_yolo_id[cat['id']] = class_name_to_yolo_id[class_name]
    
    print(f"클래스 ID 매핑 (COCO ID → YOLO ID): {coco_id_to_yolo_id}")
    
    # 1. COCO annotation JSON 파일 수정
    for split in ['train', 'val']:
        json_path = annotations_dir / f"{split}.json"
        if not json_path.exists():
            continue
        
        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # categories 필터링 및 재매핑
        new_categories = []
        for cat in coco_data['categories']:
            class_name = cat['name']
            if class_name.startswith('b_') or class_name.startswith('l_'):
                class_name = class_name[2:]
            
            if class_name in class_name_to_yolo_id:
                new_cat = cat.copy()
                new_cat['id'] = class_name_to_yolo_id[class_name]
                new_cat['name'] = class_name
                new_categories.append(new_cat)
        
        # annotations의 category_id 재매핑
        valid_annotations = []
        for ann in coco_data['annotations']:
            old_id = ann['category_id']
            if old_id in coco_id_to_yolo_id:
                ann['category_id'] = coco_id_to_yolo_id[old_id]
                valid_annotations.append(ann)
        
        # 매핑된 annotation만 유지
        coco_data['annotations'] = valid_annotations
        coco_data['categories'] = sorted(new_categories, key=lambda x: x['id'])
        
        # JSON 파일 저장
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        
        print(f"{split}.json: {len(coco_data['annotations'])}개 annotation, {len(coco_data['categories'])}개 클래스")
    
    # 2. YOLO 라벨 파일(.txt) 수정
    # 라벨 파일의 클래스 ID는 원본 COCO ID일 수 있으므로 동일한 매핑 사용
    for split in ['train', 'val']:
        labels_dir = Path(dataset_dir) / split / "labels"
        if not labels_dir.exists():
            continue
        
        label_files = list(labels_dir.glob("*.txt"))
        modified_count = 0
        skipped_count = 0
        
        for label_path in label_files:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                if not line.strip():
                    continue
                
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                old_class_id = int(parts[0])
                
                # COCO ID를 YOLO ID로 매핑
                if old_class_id in coco_id_to_yolo_id:
                    parts[0] = str(coco_id_to_yolo_id[old_class_id])
                    new_lines.append(' '.join(parts) + '\n')
                else:
                    # 매핑되지 않은 클래스는 건너뜀
                    skipped_count += 1
            
            # 파일 저장
            if new_lines:
                with open(label_path, 'w') as f:
                    f.writelines(new_lines)
                modified_count += 1
            else:
                # 라벨이 없으면 빈 파일로 유지
                label_path.touch()
        
        print(f"{split}/labels: {modified_count}개 파일 수정 완료, {skipped_count}개 라벨 건너뜀")
    
    return True


def create_data_yaml(dataset_dir: str) -> bool:
    """COCO annotation 파일에서 클래스 정보를 추출하여 data.yaml 파일을 생성합니다.
    
    Args:
        dataset_dir: 데이터셋 디렉토리 경로
    
    Returns:
        data.yaml 생성 성공 여부
    """
    annotations_dir = Path(dataset_dir) / "annotations"
    train_json_path = annotations_dir / "train.json"
    val_json_path = annotations_dir / "val.json"
    
    # train.json 또는 val.json에서 클래스 정보 추출
    annotation_file = None
    if train_json_path.exists():
        annotation_file = train_json_path
    elif val_json_path.exists():
        annotation_file = val_json_path
    
    if not annotation_file:
        print(f"경고: train.json 또는 val.json 파일을 찾을 수 없습니다. data.yaml 생성을 건너뜁니다.")
        return False
    
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # categories에서 클래스 정보 추출 (ID 순서대로 정렬)
    if 'categories' not in coco_data:
        print(f"경고: {annotation_file}에 categories 정보가 없습니다. data.yaml 생성을 건너뜁니다.")
        return False
    
    # remap_class_ids()에서 이미 재매핑된 categories를 사용
    # ID는 이미 0, 1, 2로 재매핑되고 name도 접두사가 제거된 상태
    # 원본 ID 1,4: b_fully_ripened, l_fully_ripened -> YOLO ID 0: fully_ripened
    # 원본 ID 2,5: b_half_ripened, l_half_ripened -> YOLO ID 1: half_ripened
    # 원본 ID 3,6: b_green, l_green -> YOLO ID 2: green
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    class_names = [cat['name'] for cat in categories]
    num_classes = len(class_names)
    
    # 절대 경로 계산
    target_path = Path(dataset_dir).resolve()
    
    # data.yaml 생성
    data_yaml_path = target_path / "data.yaml"
    data_yaml_content = {
        'path': str(target_path),
        'train': 'train/images',
        'val': 'val/images',
        'names': class_names,
        'nc': num_classes
    }
    
    with open(data_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"data.yaml 파일 생성 완료: {data_yaml_path}")
    print(f"  - 클래스 개수: {num_classes}")
    print(f"  - 클래스 목록: {', '.join(class_names)}")
    return True


def cleanup_cache():
    """기존 kagglehub 캐시 디렉토리를 삭제합니다."""
    cache_path = os.path.expanduser("~/.cache/kagglehub")
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
        print(f"기존 kagglehub 캐시 디렉토리 삭제: {cache_path}")
    else:
        print(f"캐시 디렉토리가 존재하지 않습니다: {cache_path}")


def main():
    """메인 함수: 데이터셋 다운로드 및 설정 전체 프로세스를 실행합니다."""
    # 설정
    dataset_name = "nexuswho/laboro-tomato"
    target_base_dir = "data/tomato_data"
    dataset_subdir = "laboro-tomato"
    
    # 1. 데이터셋 다운로드
    source_path = download_dataset(dataset_name)
    
    # 2. 데이터셋 이동
    target_dir = move_dataset(source_path, target_base_dir, dataset_subdir)
    
    if target_dir is None:
        print("데이터셋 이동 실패. 프로세스를 종료합니다.")
        return
    
    # 3. test.json을 val.json으로 이름 변경
    rename_test_to_val(target_dir)
    
    # 4. 클래스 ID 재매핑 (COCO JSON + YOLO 라벨 파일)
    remap_class_ids(target_dir)
    
    # 5. data.yaml 파일 생성
    create_data_yaml(target_dir)
    
    # 6. 캐시 삭제
    cleanup_cache()
    
    print("\n데이터셋 다운로드 및 설정이 완료되었습니다.")


if __name__ == "__main__":
    main()
