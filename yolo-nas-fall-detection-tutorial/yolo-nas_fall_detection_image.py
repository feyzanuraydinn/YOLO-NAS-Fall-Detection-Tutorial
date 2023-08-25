# Import necessary libraries and modules
import torch
from super_gradients.training import models

# Load the YOLO-NAS model
yolo_nas_l = models.get("yolo_nas_l", pretrained_weights="coco")

# Perform object detection on an image
url = "https://example.com/path/to/your/image.jpg"
result = yolo_nas_l.predict(url, conf=0.25)
result.show()