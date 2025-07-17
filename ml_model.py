# ml_model.py
import cv2
import numpy as np
import random
import os

class SafetyComplianceModel:
    def __init__(self):
        print("Dummy SafetyComplianceModel initialized.")
        self.categories = ["PPE_Violation", "Unsafe_Posture", "Restricted_Area_Breach", "Clear_Pathway_Obstruction"]

    def predict(self, image_path: str):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}. Check file path and integrity.")

            h, w, _ = image.shape
            num_detections = random.randint(0, 3)
            detections = []

            if num_detections > 0:
                for _ in range(num_detections):
                    category = random.choice(self.categories)
                    confidence = round(random.uniform(0.6, 0.95), 2)
                    x1 = random.randint(0, w - 50)
                    y1 = random.randint(0, h - 50)
                    x2 = random.randint(x1 + 30, w)
                    y2 = random.randint(y1 + 30, h)
                    bbox = [x1, y1, x2, y2]

                    detections.append({
                        "category": category,
                        "confidence": confidence,
                        "bbox": bbox,
                        "description": f"Detected {category} with confidence {confidence} in area {bbox}"
                    })

            print(f"Simulated {len(detections)} detections for {image_path}")
            return {"detections": detections, "image_dimensions": {"width": w, "height": h}}

        except Exception as e:
            print(f"Error during dummy prediction for {image_path}: {e}")
            return {"error": str(e), "detections": []}

if __name__ == "__main__":
    model = SafetyComplianceModel()
    dummy_img_path = "data/raw/test_dummy_image_for_ml.jpg"
    if not os.path.exists("data/raw"):
        os.makedirs("data/raw")
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(dummy_img_path, dummy_img)

    print(f"Testing dummy model with {dummy_img_path}")
    results = model.predict(dummy_img_path)
    print("\nDummy Model Prediction Results:")
    print(results)
    os.remove(dummy_img_path)
    print(f"Cleaned up {dummy_img_path}")