# Traffic Monitoring and Vehicle Analysis System

The Traffic Monitoring and Vehicle Analysis System is an advanced real-time solution for monitoring traffic flow, detecting vehicles, and analyzing key information such as vehicle types, license plates, and speeds. This system leverages advanced computer vision technologies like YOLO and EasyOCR, as well as object tracking algorithms. It included object detection and optical character recognition (OCR), to process video footage in real time.

## Features

1. **Vehicle Detection**
   - Detects vehicles such as cars, motorbikes, trucks, and buses using YOLOv8.

2. **License Plate Recognition**
   - Extracts license plate numbers using EasyOCR.

3. **Speed Calculation**
   - Computes vehicle speed in kilometers per hour based on object tracking.

4. **Tracking Across Frames**
   - Assigns unique IDs to vehicles for tracking across video frames.

5. **Result Export**
   - Outputs results in an Excel file, including:
     - Vehicle ID
     - License Plate
     - Confidence Score
     - Vehicle Type
     - Speed (km/h)

---

## Installation

### Prerequisites

Ensure the following are installed:

- Python 3.8+
- pip (Python package manager)

### Running the Application
1. Clone the project repository to your local machine:
- git clone https://github.com/your-repo/traffic-monitoring-system.git
- cd traffic-monitoring-system
2. Install the required Python packages:
- pip install -r requirements.txt
3. Start the Flask application:
- python traffic.py
<img width="332" alt="image" src="https://github.com/user-attachments/assets/8b93be9e-d41a-4a9e-9832-e28413360dc7" />
4. Access the API at `http://localhost:5000/process_video`.
<img width="753" alt="image" src="https://github.com/user-attachments/assets/4b0300b1-0f60-4098-ae0b-a4da9657b15a" />

### API Endpoints
Results include a video with bouding boxes:


https://github.com/user-attachments/assets/be5f1af0-d014-4080-a219-735f987b179a



and an excel file to storage results:

<img width="308" alt="image" src="https://github.com/user-attachments/assets/0feb412f-0273-48f0-9764-a994a8aa8499" />


## Directory Structure

<img width="371" alt="image" src="https://github.com/user-attachments/assets/207d4664-b4a2-4694-b54e-f872069264b1" />


## Enhancements and Customization
- **Extend Vehicle Classes:** Add more vehicle types by fine-tuning the YOLO model.
- **Speed Calibration:** Adjust the `pixels_per_meter` value for accurate speed measurements based on the camera setup.
- **Localization:** Modify EasyOCR settings to support additional languages for license plate recognition.

## Acknowledgments

- [YOLO](https://github.com/ultralytics/yolov8) for object detection
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for text recognition
