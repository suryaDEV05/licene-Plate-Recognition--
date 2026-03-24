# modules/module2_video_gpu.py
import cv2
import numpy as np
import time
import threading
from .utils import load_yolo, load_ocr, save_plate_csv, now_ts, ensure_folder

def _show_window_loop(win_name, frame_getter, stop_flag):
    """
    thread helper - continuously show latest frame from frame_getter()
    frame_getter() should return current frame or None
    """
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    while not stop_flag['stop']:
        frame = frame_getter()
        if frame is not None:
            cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag['stop'] = True
            break
    cv2.destroyAllWindows()

def load_video(video_path, yolo_path, csv_out="output/detected_plates.csv",
               excel_out="output/detected_plates.xlsx", device=0,
               ocr_gpu=True, yolo_conf_thresh=0.4, ocr_conf_thresh=0.5):
    """
    Run LPR on video using GPU.
    - device: 0 for first GPU, 'cpu' to force CPU
    - ocr_gpu: True to use EasyOCR GPU
    """
    ensure_folder("output")
    yolo = load_yolo(yolo_path, device=device)
    reader = load_ocr(gpu=ocr_gpu)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Could not open video. Check the path: {video_path}")

    frame_lock = threading.Lock()
    latest_frame = {'img': None}
    stop_flag = {'stop': False}
    frame_idx = 0
    fps_smooth = 0.0
    alpha = 0.85

    def get_frame():
        with frame_lock:
            return latest_frame['img']

    # start display thread
    t = threading.Thread(target=_show_window_loop, args=("LPR-GPU", get_frame, stop_flag), daemon=True)
    t.start()

    print("▶ Processing video on GPU... Press 'q' in window to stop.")
    csv_path = csv_out

    while not stop_flag['stop']:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        start = time.time()
        # YOLO inference on GPU (device arg ensures GPU)
        # ultralytics allows device at call: device=0
        results = yolo(frame, device=device)
        # FPS smoothing
        step = time.time() - start
        fps = 1.0 / step if step>0 else 0
        fps_smooth = alpha*fps_smooth + (1-alpha)*fps if fps_smooth>0 else fps

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            yolo_conf = float(box.conf[0])
            if yolo_conf < yolo_conf_thresh:
                continue

            # safe crop
            h,w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # OCR (EasyOCR GPU)
            ocr_results = reader.readtext(crop)
            # pick best OCR result (first) if exists and above threshold
            if len(ocr_results) == 0:
                continue
            text, ocr_conf = ocr_results[0][1], ocr_results[0][2]
            if (ocr_conf is None) or (ocr_conf < ocr_conf_thresh):
                continue
            plate = "".join(text.split())
            print("Detected Plate:", plate)

            # annotate
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame, f"{plate} {yolo_conf:.2f}/{ocr_conf:.2f}",
                        (x1, max(15,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # save CSV row
            row = {
                'timestamp': now_ts(),
                'plate': plate,
                'yolo_conf': f"{yolo_conf:.3f}",
                'ocr_conf': f"{ocr_conf:.3f}",
                'frame_idx': frame_idx
            }
            try:
                save_plate_csv(csv_path, row)
            except Exception as e:
                print("Warning: could not save CSV:", e)

        # overlay FPS
        cv2.putText(frame, f"FPS: {fps_smooth:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        # publish latest frame for display thread
        with frame_lock:
            latest_frame['img'] = frame.copy()

    # cleanup
    stop_flag['stop'] = True
    t.join(timeout=2.0)
    cap.release()
    print("✔ Finished. CSV saved to:", csv_path)
    # convert CSV->Excel
    try:
        from .utils import save_to_excel
        ok = save_to_excel(csv_path, excel_out)
        if ok:
            print("✔ Excel exported to:", excel_out)
    except Exception as e:
        print("Could not export excel:", e)
