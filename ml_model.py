# ml_model.py
# Located in: E:/Junaid New/Disrupt Labs/My practice Projects/Proactive Safety Compliance & Anomaly Detection API

import cv2
import numpy as np
import time
import logging # Import logging for consistent output

# NEW: Import the YOLO model from ultralytics
from ultralytics import YOLO

# Configure logging for ml_model.py (optional, but good practice for internal logging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SafetyComplianceModel:
    def __init__(self):
        # NEW: Load a pre-trained YOLOv8nano model (n for nano, s for small, etc.)
        # This model file (.pt) will be automatically downloaded the first time.
        self.model = YOLO('yolov8n.pt')
        logging.info("YOLOv8n model loaded successfully for SafetyComplianceModel.")
        # Define the class names the YOLO model can detect
        # These are for COCO dataset, which yolov8n.pt is trained on
        # We'll specifically look for 'person' (class ID 0)
        self.class_names = self.model.names


    def analyze_image(self, image_np_array):
        if not isinstance(image_np_array, np.ndarray):
            logging.error("Input to analyze_image must be a NumPy array.")
            return {"error": "Input must be a NumPy array (image frame)."}

        # Perform inference on the image frame
        # conf=0.5 means only show detections with confidence > 50%
        # verbose=False suppresses detailed output from the model's predict method
        results = self.model.predict(image_np_array, conf=0.5, verbose=False)

        detections = []
        compliance_status = "Compliant"
        found_person = False # Flag to track if a person is detected

        # Process the results. 'results' is a list of Results objects (one per image)
        for r in results:
            # 'boxes' contains bounding box information
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # Get integer coordinates
                confidence = round(float(box.conf[0]), 2) # Confidence score
                class_id = int(box.cls[0]) # Class ID

                label = self.class_names[class_id] # Get the class name from ID

                # Only add to detections if it's a person
                if label == "person":
                    found_person = True
                    detections.append({
                        "label": label,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2]
                    })

        # Based on detection results, determine compliance status
        # For this example, let's say "Non-Compliant" if a person is detected
        # This logic can be expanded for PPE (e.g., if person detected BUT no hardhat)
        if found_person:
            compliance_status = "Person Detected" # Or "Non-Compliant" if person in restricted area
        else:
            compliance_status = "No Person Detected" # Or "Compliant" if no person when expected

        final_results = {
            "compliance_status": compliance_status,
            "detections": detections,
            "timestamp": time.time(),
            "image_dimensions": {"width": image_np_array.shape[1], "height": image_np_array.shape[0]}
        }
        logging.info(f"ML model processed frame. Status: {compliance_status}")
        return final_results

# Optional: Keep this function if your original /analyze_safety endpoint
# needs to load images from file paths before passing to analyze_image.
def load_image_from_file(image_path):
    if not os.path.exists(image_path):
        logging.error(f"Image file not found at {image_path}")
        return None
    return cv2.imread(image_path)