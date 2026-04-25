# Train YOLOv11 on Grape.v1i.yolov11 (single experiment; same code as main.py subset)
from main import run_pipeline

if __name__ == "__main__":
    run_pipeline(only_keys=["yolov11"])
