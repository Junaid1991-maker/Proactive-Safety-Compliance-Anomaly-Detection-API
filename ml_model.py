# ml_model.py
# Located in: E:/Junaid New/Disrupt Labs/My practice Projects/Proactive Safety Compliance & Anomaly Detection API

import cv2
import numpy as np
import time
import logging

# Import the YOLO model from ultralytics
from ultralytics import YOLO

# Configure logging for ml_model.py
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SafetyComplianceModel:
    def __init__(self):
        # Model for general object detection (e.g., 'person')
        self.person_model = YOLO('yolov8n.pt')
        logging.info("YOLOv8n model loaded successfully for person detection.")

        # --- LOAD YOUR DEDICATED PPE MODEL HERE ---
        # Make sure 'ppe.pt' is the correct path to your downloaded PPE model.
        # This model should be trained to detect specific PPE items like 'hardhat', 'vest', etc.
        self.ppe_model = YOLO('./ppe.pt') # <--- IMPORTANT: This path now points to your downloaded ppe.pt
        logging.info("Dedicated PPE model loaded.")

        # Define the required PPE classes for compliance checking
        # These names MUST exactly match the class names in your PPE model's 'names' list.
        # For the ppe.pt model from Vinayakmane47's repo, common classes include 'hardhat', 'vest', 'mask', 'glove'.
        self.REQUIRED_PPE = {
            "hardhat": True, # Set to True if hard hat is required
            "vest": False    # Set to True if safety vest is required (change this to True if you want to check for vests)
            # Add other required PPE here as needed (e.g., "mask": True)
        }
        logging.info(f"Configured required PPE: {self.REQUIRED_PPE}")


    def analyze_image(self, image_np_array):
        if not isinstance(image_np_array, np.ndarray):
            logging.error("Input to analyze_image must be a NumPy array.")
            return {"error": "Input must be a NumPy array (image frame)."}

        detections = []
        compliance_status = "Compliant: No Person Detected" # Default status

        # --- Stage 1: Detect Persons ---
        # We use a slightly lower confidence threshold for initial person detection to ensure we catch everyone.
        person_results = self.person_model.predict(image_np_array, conf=0.4, verbose=False) # Reduced conf for person detection
        
        found_person = False
        non_compliant_persons_count = 0

        for r in person_results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                label = self.person_model.names[class_id]
                confidence = round(float(box.conf[0]), 2)

                if label == "person":
                    x1_p, y1_p, x2_p, y2_p = map(int, box.xyxy[0])
                    
                    # Ensure bounding box coordinates are within image dimensions
                    x1_p = max(0, x1_p)
                    y1_p = max(0, y1_p)
                    x2_p = min(image_np_array.shape[1], x2_p)
                    y2_p = min(image_np_array.shape[0], y2_p)

                    # Crop the Region of Interest (ROI) for PPE analysis
                    person_roi = image_np_array[y1_p:y2_p, x1_p:x2_p]
                    
                    person_has_all_required_ppe = True
                    missing_ppe_items = []

                    if person_roi.size > 0 and person_roi.shape[0] > 0 and person_roi.shape[1] > 0: # Ensure ROI is valid
                        # --- Stage 2: Analyze PPE within the detected person's ROI ---
                        ppe_results = self.ppe_model.predict(person_roi, conf=0.5, verbose=False) # Adjust conf for PPE model as needed

                        detected_ppe_labels = set()
                        for p_r in ppe_results:
                            for p_box in p_r.boxes:
                                ppe_class_id = int(p_box.cls[0])
                                ppe_label = self.ppe_model.names[ppe_class_id]
                                detected_ppe_labels.add(ppe_label)

                                # Add PPE detection to the overall list (adjusting coords to original image)
                                # Only add specific PPE detections, not 'no_hardhat' if that's a class
                                if ppe_label in self.REQUIRED_PPE and self.REQUIRED_PPE[ppe_label]:
                                    x1_ppe, y1_ppe, x2_ppe, y2_ppe = map(int, p_box.xyxy[0])
                                    detections.append({
                                        "label": ppe_label,
                                        "confidence": round(float(p_box.conf[0]), 2),
                                        "bbox": [x1_ppe + x1_p, y1_ppe + y1_p, x2_ppe + x1_p, y2_ppe + y1_p]
                                    })
                        
                        # Check for missing required PPE
                        for required_item, is_required in self.REQUIRED_PPE.items():
                            if is_required and required_item not in detected_ppe_labels:
                                person_has_all_required_ppe = False
                                missing_ppe_items.append(required_item)
                                logging.info(f"Person at [{x1_p},{y1_p},{x2_p},{y2_p}] missing: {required_item}")
                                
                    else:
                        logging.warning(f"Empty or invalid ROI for person at [{x1_p},{y1_p},{x2_p},{y2_p}]. Skipping PPE check.")
                        person_has_all_required_ppe = False # Cannot confirm PPE if ROI is bad
                        missing_ppe_items.append("PPE check failed (invalid ROI)")


                    # Add the person detection itself to ensure the person is always drawn
                    detections.append({
                        "label": label, # 'person'
                        "confidence": confidence,
                        "bbox": [x1_p, y1_p, x2_p, y2_p]
                    })
                    
                    found_person = True

                    # Update overall compliance status for this person
                    if not person_has_all_required_ppe:
                        non_compliant_persons_count += 1
                        # Create a specific anomaly label for visualization
                        anomaly_label = f"Non-Compliant: No {', No '.join([item.replace('_', ' ').title() for item in missing_ppe_items])}"
                        detections.append({
                            "label": anomaly_label,
                            "confidence": 1.0, # High confidence for anomaly if missing PPE
                            "bbox": [x1_p, y1_p, x2_p, y2_p] # Anomaly bounding box around the person
                        })
                        
        # Final compliance status based on all persons in the frame
        if found_person:
            if non_compliant_persons_count > 0:
                compliance_status = f"Non-Compliant: {non_compliant_persons_count} person(s) missing PPE"
            else:
                compliance_status = "Compliant: All persons wearing required PPE"
        else:
            compliance_status = "Compliant: No Person Detected"

        final_results = {
            "compliance_status": compliance_status,
            "detections": detections,
            "timestamp": time.time(),
            "image_dimensions": {"width": image_np_array.shape[1], "height": image_np_array.shape[0]}
        }
        logging.info(f"ML model processed frame. Status: {compliance_status}")
        return final_results