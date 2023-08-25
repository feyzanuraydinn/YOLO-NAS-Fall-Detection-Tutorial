# Import necessary libraries and modules
import torch
from super_gradients.training import models

# Load the YOLO-NAS model
yolo_nas_l = models.get("yolo_nas_l", pretrained_weights="coco")

# Real-time detection using webcam
model = models.get("yolo_nas_l", pretrained_weights="coco")
device = 'cuda' if torch.cuda.is_available() else "cpu"
model = model.to(device)

model.predict_webcam()