# YOLO-NAS Object Detection Tutorial

In this tutorial, we will learn how to perform object detection using the SuperGradients library with the YOLO-NAS model. We will go through the process of setting up the environment, loading the model, and detecting objects in images, videos, and even through your webcam.

## Prerequisites

Before you begin, make sure you have the following installed:

- Python 3.6+
- Visual Studio Code (or any preferred code editor)
- Pip package manager

## Step 1: Setting Up the Environment (Optional)

1. Create a new project folder for this tutorial.

2. Open a terminal/command prompt in the project folder and create a virtual environment:

    - Windows:
        ```bash
        python -m venv venv
        ```

    - macOS and Linux:
        ```bash
        python3 -m venv venv
        ```

3. Activate the virtual environment:

    - Windows:
        ```bash
        venv\Scripts\activate
        ```

    - macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

## Step 2: Installing SuperGradients

1. Install required libraries using pip:

    ```bash
    pip install super-gradients==3.1.3 imutils roboflow pytube torch
    ```

## Step 3: Writing the Code

1. Create a new Python script file in your project folder, e.g., `object_detection.py`.

2. Import necessary libraries and modules:

    ```python
    import torch
    from super_gradients.training import models
    ```

3. Load the YOLO-NAS model:

    ```python
    yolo_nas_l = models.get("yolo_nas_l", pretrained_weights="coco")
    ```

4. Perform object detection on an image:

    ```python
    url = "https://example.com/path/to/your/image.jpg"
    result = yolo_nas_l.predict(url, conf=0.25)
    result.show()
    ```

5. Detect objects in a video:

    ```python
    input_video_path = "input_video.mp4"
    output_video_path = "output_video.mp4"
    device = 'cuda' if torch.cuda.is_available() else "cpu"

    yolo_nas_l.to(device).predict(input_video_path).save(output_video_path)
    ```

6. Real-time detection using webcam:

    ```python
    model = models.get("yolo_nas_l", pretrained_weights="coco")
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    model.predict_webcam()
    ```



