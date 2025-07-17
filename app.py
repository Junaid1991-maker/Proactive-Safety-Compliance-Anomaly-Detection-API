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

import logging # NEW: Import the logging module

# NEW: Configure logging
# This sets up basic logging to the console.
# level=logging.INFO means it will log INFO, WARNING, ERROR, and CRITICAL messages.
# format defines the output format for each log entry, including timestamp and level.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file (like PORT)
load_dotenv()

app = Flask(__name__)

# Configure the upload folder for images sent to the API
app.config['UPLOAD_FOLDER'] = 'uploads'
# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize your ML model instance globally.
ml_model = SafetyComplianceModel()
logging.info("Flask app initialized and ML model loaded.") # MODIFIED: Using logging

# Global variables to control the streaming loop and camera object
streaming_active = False
camera_feed = None # To hold the cv2.VideoCapture object

# Generator function for video streaming
def generate_frames():
    global streaming_active, camera_feed
    camera_feed = cv2.VideoCapture(0)

    if not camera_feed.isOpened():
        logging.error("Could not open webcam for streaming. Check connection, permissions, and if in use.") # MODIFIED: Using logging
        streaming_active = False
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n'
               b'Error: Could not open webcam. Please check its connection, permissions, and if it\'s in use by another application.\r\n')
        return

    logging.info("Webcam successfully opened for streaming.") # MODIFIED: Using logging
    while streaming_active:
        ret, frame = camera_feed.read()
        if not ret:
            logging.warning("Failed to grab frame from stream. Ending stream.") # MODIFIED: Using logging
            break

        analysis_results = ml_model.analyze_image(frame)

        if "detections" in analysis_results and analysis_results["detections"]:
            for det in analysis_results["detections"]:
                if "bbox" in det and len(det["bbox"]) == 4:
                    x1, y1, x2, y2 = det["bbox"]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, det["label"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        status_text = f"Status: {analysis_results.get('compliance_status', 'N/A')}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    if camera_feed:
        camera_feed.release()
        logging.info("Webcam released.") # MODIFIED: Using logging
    streaming_active = False

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
        logging.info("Received request to start video stream...") # MODIFIED: Using logging
        streaming_active = True
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return jsonify({"status": "Streaming already active"}), 200

@app.route('/stop_video_feed')
def stop_video_feed():
    global streaming_active
    if streaming_active:
        logging.info("Received request to stop video stream...") # MODIFIED: Using logging
        streaming_active = False
        return jsonify({"status": "Streaming stop request received. Please allow a moment for camera to release."})
    else:
        return jsonify({"status": "No active stream to stop"})

@app.route('/analyze_safety', methods=['POST'])
def analyze_safety():
    """
    Endpoint for uploading an image file and getting safety analysis results.
    The image is processed by the ML model.
    """
    if 'file' not in request.files:
        logging.warning("No file part in the analyze_safety request.") # MODIFIED: Using logging
        return jsonify({"error": "No file part in the request. Please send an image file under the key 'file'."}), 400

    file = request.files['file']

    if file.filename == '':
        logging.warning("No selected file in the analyze_safety request.") # MODIFIED: Using logging
        return jsonify({"error": "No selected file. Please choose an image file to upload."}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            logging.info(f"Received and saved file: {filepath}") # MODIFIED: Using logging

            image_np = cv2.imread(filepath)
            if image_np is None:
                logging.error(f"Could not load image file from path: {filepath}. Ensure it's a valid image format.") # MODIFIED: Using logging
                return jsonify({"error": "Could not load image file from path. Ensure it's a valid image format."}), 400

            prediction_results = ml_model.analyze_image(image_np)
            logging.info(f"Prediction results for {filename}: %s", prediction_results) # MODIFIED: Using logging

            os.remove(filepath)
            logging.info(f"Deleted temporary file: {filepath}") # MODIFIED: Using logging

            return jsonify({
                "message": "Image analyzed successfully!",
                "original_filename": file.filename,
                "analysis_results": prediction_results
            }), 200
        except Exception as e:
            logging.error(f"An error occurred during file processing for {filename}: {str(e)}", exc_info=True) # MODIFIED: Using logging with exc_info=True
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
        logging.error(f"Error serving uploaded file {filename}: {str(e)}", exc_info=True) # MODIFIED: Using logging
        return jsonify({"error": f"File not found or access denied: {str(e)}"}), 404


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)