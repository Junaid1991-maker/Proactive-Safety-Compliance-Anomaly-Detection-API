# ml_model.py
# Located in: E:/Junaid New/Disrupt Labs/My practice Projects/Proactive Safety Compliance & Anomaly Detection API

import cv2
import numpy as np
import os
import time
import random # NEW: Import the random module

class SafetyComplianceModel:
    def __init__(self):
        print("Dummy SafetyComplianceModel initialized.") # Can change to logging.info if you want

    def analyze_image(self, image_np_array):
        if not isinstance(image_np_array, np.ndarray):
            print("Error: Input to analyze_image must be a NumPy array.")
            return {"error": "Input must be a NumPy array (image frame)."}

        # Simulate analysis time for a realistic feel
        time.sleep(0.1) # Simulate a 100ms processing time

        height, width, _ = image_np_array.shape
        detections = []
        compliance_status = "Compliant"

        # NEW/MODIFIED: More advanced dummy detection logic
        # Simulate a chance of non-compliance and varying detections
        if random.random() < 0.3: # 30% chance of detecting a non-compliance/anomaly
            compliance_status = "Non-Compliant"
            
            # Randomly pick a "violation" type
            violation_type = random.choice([
                "No Hard Hat",
                "No Safety Vest",
                "Unsafe Area Entry",
                "Equipment Misplaced",
                "Trip Hazard Detected"
            ])
            
            # Generate random bounding box coordinates (ensure they are within image bounds)
            # Make sure bbox is not outside image dimensions
            x_min = random.randint(0, width - 150)
            y_min = random.randint(0, height - 150)
            x_max = random.randint(x_min + 50, min(x_min + 300, width))
            y_max = random.randint(y_min + 50, min(y_min + 300, height))

            detections.append({
                "label": violation_type,
                "confidence": round(random.uniform(0.7, 0.95), 2), # Random confidence score
                "bbox": [x_min, y_min, x_max, y_max]
            })

        else: # 70% chance of being compliant
            compliance_status = "Compliant"
            detections.append({
                "label": "All Clear - Compliant",
                "confidence": 0.99,
                "bbox": []
            })

        results = {
            "compliance_status": compliance_status,
            "detections": detections,
            "timestamp": time.time(),
            "image_dimensions": {"width": width, "height": height}
        }
        print(f"Dummy model processed frame. Status: {compliance_status}") # Can change to logging.info
        return results

# Optional: Keep this function if your original /analyze_safety endpoint
# needs to load images from file paths before passing to analyze_image.
def load_image_from_file(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}") # Can change to logging.error
        return None
    return cv2.imread(image_path)