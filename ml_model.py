# ml_model.py
# Located in: E:/Junaid New/Disrupt Labs/My practice Projects/Proactive Safety Compliance & Anomaly Detection API

import cv2
import numpy as np
import time
import logging
import os
import datetime
import requests # Make sure this is imported at the top

# Import the YOLO model from ultralytics
from ultralytics import YOLO

# Configure logging for ml_model.py
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SafetyComplianceModel:
    def __init__(self):
        logging.info("Initializing SafetyComplianceModel...")

        # --- Handle yolov8n.pt model ---
        # YOLOv8 often downloads 'yolov8n.pt' automatically on first use if not found.
        # If Heroku build is failing due to yolov8n.pt not found, you might need a download URL for it too.
        # For this example, we assume it's handled by ultralytics or is present.
        self.person_model_path = './yolov8n.pt'
        try:
            self.person_model = YOLO(self.person_model_path)
            logging.info(f"YOLOv8n model loaded successfully from {self.person_model_path}.")
        except Exception as e:
            logging.error(f"Failed to load yolov8n.pt model from {self.person_model_path}: {e}")
            raise RuntimeError(f"Critical error: Failed to load YOLOv8n model from {self.person_model_path}. Make sure it's accessible or add specific download logic for it if needed for Heroku.")


        # --- Handle Dedicated PPE Model (ppe.pt) ---
        self.ppe_model_path = './ppe.pt'
        # URL to download your ppe.pt from (raw.githubusercontent.com is good for this)
        self.ppe_model_url = "https://raw.githubusercontent.com/Vinayakmane47/PPE_detection_YOLO/main/YOLO-Weights/ppe.pt" # YOUR MODEL URL

        if not os.path.exists(self.ppe_model_path):
            logging.info(f"PPE model not found at {self.ppe_model_path}. Attempting to download from {self.ppe_model_url}...")
            try:
                response = requests.get(self.ppe_model_url, stream=True, timeout=30) # Added timeout
                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
                with open(self.ppe_model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logging.info("PPE model downloaded successfully!")
            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to download PPE model from {self.ppe_model_url}: {e}")
                raise RuntimeError(f"Critical error: Could not download PPE model: {e}. Check URL or network connectivity.")
            except Exception as e:
                logging.error(f"An unexpected error occurred during PPE model download: {e}")
                raise RuntimeError(f"Critical error: An unexpected error occurred during PPE model download: {e}")
        else:
            logging.info(f"PPE model found locally at {self.ppe_model_path}.")

        self.ppe_model = YOLO(self.ppe_model_path)
        logging.info("Dedicated PPE model loaded.")

        # --- Configuration for Anomaly Logging ---
        self.log_file_path = os.path.join('anomaly_logs', 'anomaly_log.txt') # Log file inside anomaly_logs folder
        self.snapshot_dir = 'anomaly_logs' # Directory for snapshots (same as log file for simplicity)
        os.makedirs(self.snapshot_dir, exist_ok=True) # Ensure the directory exists
        logging.info(f"Anomaly logs will be saved to: {self.log_file_path} and snapshots to: {self.snapshot_dir}")

        # Keep track of the last logged anomaly time to prevent rapid re-logging
        self.last_logged_anomaly_time = 0
        self.logging_cooldown_seconds = 10 # Only log a new anomaly every 10 seconds per person
        logging.info("SafetyComplianceModel initialization complete.")

    def _log_anomaly(self, frame, person_bbox, missing_ppe_items, person_id=None):
        """
        Internal method to log an anomaly and save a snapshot.
        """
        current_time = time.time()

        # Add a cooldown mechanism
        if (current_time - self.last_logged_anomaly_time) < self.logging_cooldown_seconds:
            logging.debug(f"Skipping anomaly log due to cooldown. Last log was {current_time - self.last_logged_anomaly_time:.2f} seconds ago.")
            return

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        anomaly_description = f"Missing: {', '.join([item.replace('_', ' ').title() for item in missing_ppe_items])}"

        # Define snapshot filename
        snapshot_filename = f"anomaly_{timestamp_str}_{person_id if person_id else 'unknown'}.jpg"
        snapshot_filepath = os.path.join(self.snapshot_dir, snapshot_filename)

        # Draw the anomaly box on the snapshot before saving
        x1, y1, x2, y2 = person_bbox
        # Ensure coordinates are within frame boundaries for drawing
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        snapshot_frame = frame.copy()
        cv2.rectangle(snapshot_frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red box
        cv2.putText(snapshot_frame, "Anomaly: " + anomaly_description, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(snapshot_frame, timestamp_str, (10, snapshot_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save snapshot
        try:
            cv2.imwrite(snapshot_filepath, snapshot_frame)
            logging.info(f"Anomaly snapshot saved: {snapshot_filepath}")
        except Exception as e:
            logging.error(f"Failed to save anomaly snapshot {snapshot_filepath}: {e}")

        # Log entry to file
        log_entry = (
            f"Timestamp: {timestamp_str}\n"
            f"Anomaly Type: Missing PPE\n"
            f"Details: {anomaly_description}\n"
            f"Person BBox: {person_bbox}\n"
            f"Snapshot: {snapshot_filepath}\n"
            f"{'-'*50}\n" # Separator for easy parsing
        )
        try:
            with open(self.log_file_path, 'a') as f:
                f.write(log_entry)
            logging.info(f"Anomaly logged to {self.log_file_path}")
            self.last_logged_anomaly_time = current_time # Update last logged time
        except Exception as e:
            logging.error(f"Failed to write anomaly log to {self.log_file_path}: {e}")

    def get_last_n_anomaly_logs(self, n=5):
        """
        Reads the last N full anomaly entries from the log file.
        Each entry is separated by a line of 50 hyphens.
        """
        log_entries = []
        try:
            with open(self.log_file_path, 'r') as f:
                lines = f.readlines()

            current_entry_lines = []
            # Iterate through lines in reverse to find full entries efficiently
            for line in reversed(lines):
                current_entry_lines.insert(0, line) # Add to the beginning to maintain original order within an entry
                if line.strip() == '-' * 50: # Check for the separator
                    log_entries.insert(0, "".join(current_entry_lines).strip()) # Add the full entry to the front of list
                    current_entry_lines = [] # Reset for next entry
                    if len(log_entries) >= n:
                        break # Stop if we have enough entries

            # If there's a partial entry at the start due to file not ending with separator
            if current_entry_lines and len(log_entries) < n:
                log_entries.insert(0, "".join(current_entry_lines).strip())

        except FileNotFoundError:
            logging.warning(f"Anomaly log file not found at {self.log_file_path}")
        except Exception as e:
            logging.error(f"Error reading anomaly log file: {e}")
        
        return log_entries[:n] # Return up to N entries (in case more partial entries were gathered)


    def analyze_image(self, image_np_array):
        if not isinstance(image_np_array, np.ndarray):
            logging.error("Input to analyze_image must be a NumPy array.")
            return {"error": "Input must be a NumPy array (image frame)."}

        detections = []
        compliance_status = "Compliant: No Person Detected" # Default status

        # --- Stage 1: Detect Persons ---
        person_results = self.person_model.predict(image_np_array, conf=0.4, verbose=False)

        found_person = False
        non_compliant_persons_count = 0
        person_id_counter = 0 # Simple counter for unique person ID within a frame

        for r in person_results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                label = self.person_model.names[class_id]
                confidence = round(float(box.conf[0]), 2)

                if label == "person":
                    person_id_counter += 1
                    current_person_id = f"person_{person_id_counter}"

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
                                # Only add specific PPE detections that are *required* and detected
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

                        # Call the logging method here
                        self._log_anomaly(image_np_array, [x1_p, y1_p, x2_p, y2_p], missing_ppe_items, current_person_id)

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