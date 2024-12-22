import os
import cv2
import numpy as np
import pandas as pd
import easyocr
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from flask import Flask, jsonify, request, send_file
from io import BytesIO

app = Flask(__name__)
# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)  # Use GPU if available, else CPU
# Function to assign object IDs across frames
def track_objects(current_boxes, prev_boxes, prev_ids):
    new_ids = []
    if len(prev_boxes) == 0:
        new_ids = list(range(1, len(current_boxes) + 1))
    else:
        if len(current_boxes) > 0 and len(prev_boxes) > 0:
            current_boxes = np.array(current_boxes).reshape(-1, 2)
            prev_boxes = np.array(prev_boxes).reshape(-1, 2)
            dist_matrix = cdist(current_boxes, prev_boxes, metric='euclidean')
            row_ind, col_ind = np.where(dist_matrix < 100)
            assigned_ids = {j: prev_ids[i] for i, j in zip(col_ind, row_ind)}
            new_ids = [assigned_ids.get(i, max(prev_ids) + 1 + j) for j, i in enumerate(range(len(current_boxes)))]
        else:
            new_ids = list(range(1, len(current_boxes) + 1))
    return new_ids
# Function to read license plate using EasyOCR
def read_license_plate(plate_image):
    try:
        result = reader.readtext(plate_image)
        if result:
            plate_text = result[0][1]
            score = result[0][2]
            return plate_text, score
        else:
            return None, 0
    except Exception as e:
        return None, 0
# Load YOLO models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')
# Initialize object tracker
prev_boxes = []
prev_ids = []
vehicles = {2: "Car", 3: "Motorbike", 5: "Truck", 7: "Bus"}
results = {}
prev_positions = {}
@app.route('/process_video', methods=['POST'])
def process_video():
    # Load video from request
    video_file = request.files['video']
    video_path = './uploaded_video.mp4'
    video_file.save(video_path)
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Cannot open video file!"}), 400
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_nmr = -1
    ret = True
    global results
    results = {}  # Reset results for each request
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_frame = clahe.apply(gray_frame)
        blurred_frame = cv2.GaussianBlur(enhanced_frame, (5, 5), 0)
        frame_resized = cv2.resize(frame, (720, 480))
        detections = coco_model(frame_resized)[0]
        current_boxes = []
        current_bboxes = []
        current_classes = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                current_boxes.append([(x1 + x2) / 2, (y1 + y2) / 2])
                current_bboxes.append([x1, y1, x2, y2])
                current_classes.append(int(class_id))

        current_ids = track_objects(current_boxes, prev_boxes, prev_ids)
        prev_boxes = current_boxes
        prev_ids = current_ids
        license_plates = license_plate_detector(frame_resized)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1_lp, y1_lp, x2_lp, y2_lp, score, class_id = license_plate
            for i, (xcar1, ycar1, xcar2, ycar2) in enumerate(current_bboxes):
                if (x1_lp >= xcar1 and x2_lp <= xcar2) and (y1_lp >= ycar1 and y2_lp <= ycar2):
                    car_id = current_ids[i]
                    vehicle_type = vehicles.get(current_classes[i], "Unknown")

                    license_plate_crop = frame_resized[int(y1_lp):int(y2_lp), int(x1_lp):int(x2_lp), :]
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)
                    speed = None
                    if car_id in prev_positions:
                        prev_position = prev_positions[car_id]
                        distance = np.linalg.norm(np.array([current_boxes[i][0], current_boxes[i][1]]) - np.array([prev_position[0], prev_position[1]]))
                        pixels_per_meter = 0.05
                        distance_in_meters = distance * pixels_per_meter
                        time = 1 / fps
                        speed = (distance_in_meters / time) / 1000 * 3600

                    if license_plate_text:
                        if car_id not in results:
                            results[car_id] = {
                                "License Plate": license_plate_text,
                                "Score": license_plate_text_score,
                                "Vehicle Type": vehicle_type,
                                "Speed (km/h)": speed if speed else 0
                            }
                        else:
                            if results[car_id]["Score"] < license_plate_text_score:
                                results[car_id] = {
                                    "License Plate": license_plate_text,
                                    "Score": license_plate_text_score,
                                    "Vehicle Type": vehicle_type,
                                    "Speed (km/h)": speed if speed else 0
                                }
                    cv2.rectangle(frame_resized, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)
                    cv2.rectangle(frame_resized, (int(x1_lp), int(y1_lp)), (int(x2_lp), int(y2_lp)), (255, 0, 0), 2)
                    if license_plate_text:
                        cv2.putText(frame_resized, license_plate_text, (int(x1_lp), int(y1_lp) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(frame_resized, vehicle_type, (int(xcar1), int(ycar1) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    prev_positions[car_id] = current_boxes[i]
        cv2.imshow('Vehicle Detection and License Plate Recognition', frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
    df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    df.columns = ['Car ID', 'License Plate', 'Score', 'Vehicle Type', 'Speed (km/h)']
    # Save the Excel file to a BytesIO object
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    return send_file(output, as_attachment=True, download_name="vehicle_monitor.xlsx", mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == '__main__':
    app.run(debug=True)
