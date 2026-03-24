# modules/utils.py
import easyocr
from ultralytics import YOLO
import csv
import pandas as pd
import os
from datetime import datetime

def load_yolo(yolo_path, device=0):
    """Load YOLO model; device=0 for GPU, device='cpu' for CPU."""
    # Passing device at inference is recommended; still load model here
    return YOLO(yolo_path)

def load_ocr(gpu=True):
    """Load EasyOCR reader. gpu=True to use GPU (requires torch with CUDA)."""
    return easyocr.Reader(['en'], gpu=gpu)

def ensure_folder(path):
    os.makedirs(path, exist_ok=True)

def save_plate_csv(csv_path, row):
    """Append a row dict to CSV, create file with header if missing."""
    ensure_folder(os.path.dirname(csv_path) or ".")
    file_exists = os.path.exists(csv_path)
    header = ['timestamp','plate','yolo_conf','ocr_conf','frame_idx']
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([row.get(h,"") for h in header])

def save_to_excel(csv_path, excel_path):
    """Convert CSV to Excel."""
    if not os.path.exists(csv_path):
        return False
    df = pd.read_csv(csv_path)
    df.to_excel(excel_path, index=False)
    return True

def now_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
