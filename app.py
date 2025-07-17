# app.py
import os
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import cv2 # Keep this import, even if not directly used in app.py logic yet
from ml_model import SafetyComplianceModel # Import your ML model class

# Load environment variables from .env file (like PORT)
load_dotenv()

app = Flask(__name__)

# Configure the upload folder for images sent to the API
app.config['UPLOAD_FOLDER'] = 'uploads'
# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize your ML model instance globally.
# In a real production system, for heavy models, consider lazy loading or
# running inference in a separate worker process to keep the API responsive.
ml_model = SafetyComplianceModel()
print("Flask app initialized and ML model loaded.")

@app.route('/')
def home():
    """
    A simple home route to confirm the API is running.
    """
    return "Welcome to the Safety Compliance AI API! Send POST requests to /analyze_safety."

@app.route('/analyze_safety', methods=['POST'])
def analyze_safety():
    """
    Endpoint for uploading an image file and getting safety analysis results.
    The image is processed by the ML model.
    """
    # Check if a file was sent in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request. Please send an image file under the key 'file'."}), 400
    
    file = request.files['file']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({"error": "No selected file. Please choose an image file to upload."}), 400
    
    # Process the file if it exists and has an allowed extension (e.g., .jpg, .png)
    if file: # You might add file type validation here
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        try:
            file.save(filename) # Save the uploaded file temporarily
            print(f"Received and saved file: {filename}")

            # Pass the saved image file path to your ML model for prediction
            prediction_results = ml_model.predict(filename)
            print(f"Prediction results for {filename}: {prediction_results}")

            # Optionally, delete the uploaded file after processing to save space
            os.remove(filename)
            print(f"Deleted temporary file: {filename}")

            return jsonify({
                "message": "Image analyzed successfully!",
                "original_filename": file.filename,
                "analysis_results": prediction_results # Contains dummy detections
            }), 200
        except Exception as e:
            # Ensure these two lines (print and return) are indented
            # exactly four spaces (one tab equivalent) from the 'except' line.
            print(f"An error occurred during file processing: {str(e)}") # This is line 50
            return jsonify({"error": f"An error occurred during file processing: {str(e)}"}), 500 # This is line 51
    
    return jsonify({"error": "Something went wrong during file upload or processing. Please try again."}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Endpoint to serve uploaded files (for verification/debugging, not typical for production API output).
    """
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        return jsonify({"error": f"File not found or access denied: {str(e)}"}), 404

if __name__ == '__main__':
    # Get port from environment variable, default to 5000 if not set
    port = int(os.environ.get('PORT', 5000))
    # Run the Flask app. debug=True allows for automatic reloading on code changes
    # and provides a debugger, but should NEVER be used in production.
    app.run(debug=True, host='0.0.0.0', port=port)