# License Plate Recognition System (LPR)

## Overview

This project is a computer vision-based License Plate Recognition (LPR) system that detects vehicle license plates and extracts text from them. It uses YOLO (Ultralytics) for object detection and EasyOCR for optical character recognition.

The system can process both images and videos to identify license plates in real-time and display the recognized text.

---

## Features

* Real-time license plate detection
* OCR-based text extraction using EasyOCR
* Supports both image and video input
* Fast and efficient detection using YOLO
* Image preprocessing with OpenCV for better accuracy

---

## Tech Stack

* Python
* OpenCV
* Ultralytics YOLO
* EasyOCR

---

## Project Structure

LPR-Project/
│── images/              # Input images
│── videos/              # Input videos
│── output/              # Output results
│── models/              # YOLO model files
│── main.py              # Main execution file
│── requirements.txt     # Dependencies
│── README.md            # Project documentation

---

## Installation

1. Clone the repository
   git clone https://github.com/your-username/lpr-project.git
   cd lpr-project

2. Install dependencies
   pip install -r requirements.txt

---

## Usage

Run on Image:
python main.py --image path_to_image.jpg

Run on Video:
python main.py --video path_to_video.mp4

---

## How It Works

1. YOLO model detects license plate region
2. Detected region is cropped
3. Preprocessing applied using OpenCV
4. EasyOCR extracts text from plate
5. Output displayed with bounding box and text

---

## Future Improvements

* Improve OCR accuracy in low-light conditions
* Add support for multiple country plate formats
* Deploy as a web application
* Integrate database for vehicle tracking

## Author

Suryateja
---

## License

This project is open-source and available under the MIT License.
