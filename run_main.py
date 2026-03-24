# run_main.py
from modules.module2_video_gpu import load_video
import argparse

parser = argparse.ArgumentParser(description="GPU YOLO + EasyOCR LPR")
parser.add_argument("--video_path", required=True, type=str, help="Path to input video file")
parser.add_argument("--yolo_path", required=True, type=str, help="Path to YOLO .pt file")
parser.add_argument("--device", required=False, default=0, help="Device index (0) or 'cpu'")
parser.add_argument("--csv_out", required=False, default="output/detected_plates.csv")
parser.add_argument("--excel_out", required=False, default="output/detected_plates.xlsx")
args = parser.parse_args()

# device can be 'cpu' or integer gpu index
device = args.device
if device != 'cpu':
    try:
        device = int(device)
    except:
        device = 0

load_video(video_path=args.video_path, yolo_path=args.yolo_path,
           csv_out=args.csv_out, excel_out=args.excel_out, device=device, ocr_gpu=True)
