# Import necessary libraries and modules
import torch
from super_gradients.training import models

# Load the YOLO-NAS model
yolo_nas_l = models.get("yolo_nas_l", pretrained_weights="coco")

# Detect objects in a video
input_video_path = "input_video.mp4"
output_video_path = "output_video.mp4"
device = 'cuda' if torch.cuda.is_available() else "cpu"

yolo_nas_l.to(device).predict(input_video_path).save(output_video_path)