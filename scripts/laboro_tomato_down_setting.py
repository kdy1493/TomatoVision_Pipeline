''' 
kagglehub에서 데이터셋 다운로드 후 data/tomato_data/laboro-tomato 에 이동
실행 명령어: python scripts/laboro_tomato_down_setting.py
test.json을 val.json으로 이름 변경
data.yaml 파일 생성
기존 kagglehub 캐시 디렉토리 삭제
3종 클래스로 변경 : green, half_ripened, fully_ripened
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
    
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    
    # 클래스 이름에서 접두사(b_, l_) 제거하고 중복 제거
    # b_fully_ripened, l_fully_ripened → fully_ripened
    # b_half_ripened, l_half_ripened → half_ripened
    # b_green, l_green → green
    # 원본 annotation의 ID 순서를 유지하여 매핑
    # 원본 ID 1: b_fully_ripened -> fully_ripened (YOLO ID 0)
    # 원본 ID 2: b_half_ripened -> half_ripened (YOLO ID 1)
    # 원본 ID 3: b_green -> green (YOLO ID 2)
    class_id_to_name = {}
    for cat in categories:
        class_name = cat['name']
        # 접두사 제거 (b_ 또는 l_)
        if class_name.startswith('b_') or class_name.startswith('l_'):
            class_name = class_name[2:]  # 앞의 2글자(b_ 또는 l_) 제거
        
        # 원본 ID를 기반으로 매핑 (b_와 l_는 같은 클래스이므로 첫 번째 ID만 사용)
        if class_name not in class_id_to_name:
            class_id_to_name[class_name] = cat['id']
    
    # 원본 annotation의 ID 순서대로 정렬 (fully_ripened, half_ripened, green)
    # ID 1, 2, 3 순서를 유지
    class_names = sorted(class_id_to_name.keys(), key=lambda x: class_id_to_name[x])
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
    
    # 4. data.yaml 파일 생성
    create_data_yaml(target_dir)
    
    # 5. 캐시 삭제
    cleanup_cache()
    
    print("\n데이터셋 다운로드 및 설정이 완료되었습니다.")


if __name__ == "__main__":
    main()
