# app.py
# Located in: E:/Junaid New/Disrupt Labs/My practice Projects/Proactive Safety Compliance & Anomaly Detection API

import os
import cv2
import threading
import time
from flask import Flask, request, jsonify, Response, render_template, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from ml_model import SafetyComplianceModel

import logging

# Configure logging
# Set to INFO for normal operation. Change to DEBUG if you want to see per-frame processing logs.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file (like PORT)
load_dotenv()

# Initialize your Flask app instance FIRST
app = Flask(__name__)

# Configure the upload folder for images sent to the API
app.config['UPLOAD_FOLDER'] = 'uploads'
# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize your ML model instance globally AFTER the app is defined.
ml_model = SafetyComplianceModel()
logging.info("Flask app initialized and ML model loaded.")

# Global variables to control the streaming loop and camera object
streaming_active = False
camera_feed = None # To hold the cv2.VideoCapture object
last_compliance_status = "N/A" # To hold the last detected status for skipped frames
# Global dictionary to store current metrics for the dashboard
current_metrics = {
    "fps": 0.0,
    "inference_time_ms": 0.0,
    "last_analysis_status": "N/A",
    "is_anomaly": False # NEW: Flag for anomaly detection
}


# Generator function for video streaming (must be defined before routes that use it, like /video_feed)
def generate_frames():
    global streaming_active, camera_feed, last_compliance_status, current_metrics # Access global metrics
    camera_feed = cv2.VideoCapture(0)

    if not camera_feed.isOpened():
        logging.error("Could not open webcam for streaming. Check connection, permissions, and if in use.")
        streaming_active = False
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n'
               b'Error: Could not open webcam. Please check its connection, permissions, and if it\'s in use by another application.\r\n')
        return

    logging.info("Webcam successfully opened for streaming.")

    frame_count = 0
    # Adjust this value: 1 means process every frame, 2 means process every 2nd frame, etc.
    # Start with 3 or 4 and increase if needed for performance.
    skip_frames_interval = 3 # Process ML on every 3rd frame
    
    # Variables for FPS calculation
    start_time = time.time()
    processed_frame_count = 0 # Count frames actually sent for ML processing

    while streaming_active:
        ret, frame = camera_feed.read()
        if not ret:
            logging.warning("Failed to grab frame from stream. Ending stream.")
            break

        # Make a copy of the frame to draw on, so the original remains untouched if needed
        processed_frame = frame.copy()

        # Initialize is_anomaly flag for this frame before processing
        current_metrics["is_anomaly"] = False 

        # Apply frame skipping logic
        if frame_count % skip_frames_interval == 0:
            # Measure inference time
            inference_start_time = time.time()
            analysis_results = ml_model.analyze_image(frame) # Pass original frame for analysis
            inference_end_time = time.time()
            
            # Update metrics
            current_metrics["inference_time_ms"] = round((inference_end_time - inference_start_time) * 1000, 2)
            processed_frame_count += 1

            last_compliance_status = analysis_results.get('compliance_status', 'N/A')
            current_metrics["last_analysis_status"] = last_compliance_status

            # --- NEW: Check if current status indicates an anomaly for alerting ---
            if "Non-Compliant" in last_compliance_status:
                current_metrics["is_anomaly"] = True
            # --- END NEW ---

            # Draw detections on the processed_frame
            if "detections" in analysis_results and analysis_results["detections"]:
                for det in analysis_results["detections"]:
                    if "bbox" in det and len(det["bbox"]) == 4:
                        x1, y1, x2, y2 = det["bbox"]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Define colors based on label
                        color = (0, 255, 0) # Default Green for general detections (like 'person' or detected PPE)
                        text_color = (0, 255, 0)

                        # Check if the label string contains "Non-Compliant" for red color
                        if "Non-Compliant" in det["label"]:
                            color = (0, 0, 255) # Red for anomaly
                            text_color = (255, 255, 255) # White text on red background for visibility
                        
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(processed_frame, det["label"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
            
            # logging.debug only if logging level is DEBUG
            logging.debug(f"Frame {frame_count} processed. Status: {last_compliance_status}, Inf. Time: {current_metrics['inference_time_ms']}ms")

        else:
            # logging.debug only if logging level is DEBUG
            logging.debug(f"Frame {frame_count} skipped ML processing. Using last status: {last_compliance_status}")

        # Always update status text on the frame sent to browser,
        # using the last known status (either current frame's or previous analyzed frame's).
        status_text = f"Status: {last_compliance_status}"
        cv2.putText(processed_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Calculate and update FPS for display on the frame itself and in metrics
        if (time.time() - start_time) > 1: # Update FPS every second
            fps = processed_frame_count / (time.time() - start_time)
            current_metrics["fps"] = round(fps, 2)
            processed_frame_count = 0
            start_time = time.time()
        
        # Display FPS and Inference Time on the frame
        fps_text = f"FPS: {current_metrics['fps']}"
        inference_time_text = f"Inf. Time: {current_metrics['inference_time_ms']}ms"
        cv2.putText(processed_frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(processed_frame, inference_time_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


        # Increment frame count
        frame_count += 1

        ret, buffer = cv2.imencode('.jpg', processed_frame) # Use processed_frame for streaming
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    if camera_feed:
        camera_feed.release()
        logging.info("Webcam released.")
    streaming_active = False
    last_compliance_status = "N/A" # Reset status when stream ends
    current_metrics["fps"] = 0.0 # Reset metrics when stream ends
    current_metrics["inference_time_ms"] = 0.0
    current_metrics["last_analysis_status"] = "N/A"
    current_metrics["is_anomaly"] = False # Reset anomaly flag on stream end


# Define all your Flask routes here, after the 'app' object is created.

@app.route('/')
def home():
    """
    A simple home route to confirm the API is running.
    """
    return "Welcome to the Safety Compliance AI API! Visit <a href='/monitor'>/monitor</a> for live stream, or send POST requests to /analyze_safety."

@app.route('/monitor')
def monitor():
    """
    Renders the HTML page for the real-time video monitor.
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global streaming_active
    if not streaming_active:
        logging.info("Received request to start video stream...")
        streaming_active = True
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return jsonify({"status": "Streaming already active"}), 200

@app.route('/stop_video_feed')
def stop_video_feed():
    global streaming_active
    if streaming_active:
        logging.info("Received request to stop video stream...")
        streaming_active = False
        return jsonify({"status": "Streaming stop request received. Please allow a moment for camera to release."})
    else:
        return jsonify({"status": "No active stream to stop"})

# This is the /metrics route for the dashboard, correctly placed after app initialization
@app.route('/metrics')
def get_metrics():
    global current_metrics
    return jsonify(current_metrics)

@app.route('/analyze_safety', methods=['POST'])
def analyze_safety():
    """
    Endpoint for uploading an image file and getting safety analysis results.
    The image is processed by the ML model.
    """
    if 'file' not in request.files:
        logging.warning("No file part in the analyze_safety request.")
        return jsonify({"error": "No file part in the request. Please send an image file under the key 'file'."}), 400

    file = request.files['file']

    if file.filename == '':
        logging.warning("No selected file in the analyze_safety request.")
        return jsonify({"error": "No selected file. Please choose an image file to upload."}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            logging.info(f"Received and saved file: {filepath}")

            image_np = cv2.imread(filepath)
            if image_np is None:
                logging.error(f"Could not load image file from path: {filepath}. Ensure it's a valid image format.")
                return jsonify({"error": "Could not load image file from path. Ensure it's a valid image format."}), 400

            prediction_results = ml_model.analyze_image(image_np)
            logging.info(f"Prediction results for {filename}: %s", prediction_results)

            os.remove(filepath)
            logging.info(f"Deleted temporary file: {filepath}")

            return jsonify({
                "message": "Image analyzed successfully!",
                "original_filename": file.filename,
                "analysis_results": prediction_results
            }), 200
        except Exception as e:
            logging.error(f"An error occurred during file processing for {filename}: {str(e)}", exc_info=True)
            return jsonify({"error": f"An error occurred during file processing: {str(e)}"}), 500

    return jsonify({"error": "Something went wrong during file upload or processing. Please try again."}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Endpoint to serve uploaded files (for verification/debugging).
    """
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logging.error(f"Error serving uploaded file {filename}: {str(e)}", exc_info=True)
        return jsonify({"error": f"File not found or access denied: {str(e)}"}), 404


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # debug=True allows for automatic reloading on code changes
    # and provides a debugger, but should NEVER be used in production.
    app.run(debug=True, host='0.0.0.0', port=port)