''' 
실행 명령어 : python scripts/3d_visualizer.py data/tomato_data/tomato_video/smartfarm_251114
'''

import autorootcwd
import argparse
from src.viewer.viewer_pipeline import ViewerPipeline

def parse_arguments():
    """명령행 인자 파싱"""
    ap = argparse.ArgumentParser(description="Preprocessed Directory → Rerun log (Class-based)")
    ap.add_argument("input_dir", type=str, help="Path to the directory from rosbag_preprocess.py")
    return ap.parse_args()

def main():
    """메인 함수"""
    args = parse_arguments()
    pipeline = ViewerPipeline(args)
    pipeline.run()

if __name__ == "__main__":
    main()
