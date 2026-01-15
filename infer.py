import os
import cv2
import torch
import numpy as np
from model.detector import Detector

# ------------------------
# Configuration
# ------------------------
IMAGE_DIR = "demo/images"
OUTPUT_DIR = "demo/output"
MODEL_PATH = "detector.pth"
IMG_SIZE = 224
DEVICE = "cpu"

CLASS_NAMES = ["person", "car", "dog"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------
# Load Model
# ------------------------
model = Detector()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ------------------------
# Inference Loop
# ------------------------
for img_name in os.listdir(IMAGE_DIR):
    img_path = os.path.join(IMAGE_DIR, img_name)

    image = cv2.imread(img_path)
    if image is None:
        continue

    h, w, _ = image.shape

    # Preprocess
    img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img_rgb / 255.0, dtype=torch.float32)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        pred_box, pred_cls = model(img_tensor)

    # Decode predictions
    box = pred_box[0].numpy()
    cls_id = torch.argmax(pred_cls[0]).item()
    label = CLASS_NAMES[cls_id]

    # Convert normalized box to image coordinates
    x_c, y_c, bw, bh = box
    x1 = int((x_c - bw / 2) * w)
    y1 = int((y_c - bh / 2) * h)
    x2 = int((x_c + bw / 2) * w)
    y2 = int((y_c + bh / 2) * h)

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image,
        label,
        (x1, max(y1 - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    # Save output
    out_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(out_path, image)

    print(f"Processed: {img_name}")

print("Inference completed.")
