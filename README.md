# Pose Estimation and Repetition Counter

This project uses OpenCV and MediaPipe to perform real-time pose estimation and count repetitions of arm movements based on the angles of the elbow joints.

## Prerequisites

- Python 3.x
- OpenCV
- MediaPipe
- NumPy

You can install the required libraries using pip:

```bash
pip install opencv-python mediapipe numpy
```
## Usage

1. Run the script using Python:
    ```bash
    python script_name.py
    ```
2. The webcam feed will be displayed with real-time annotations showing the angles of the arms and the repetition counts.

3. Press the 'q' key to exit the application.

## How It Works

- The script uses MediaPipe's pose estimation model to detect the coordinates of key landmarks on the body.
- The angles at the elbows are calculated using the coordinates of the shoulder, elbow, and wrist.
- The stage of the arm movement (up or down) is determined based on the angle.
- Repetition counts are incremented when the arm transitions from the "down" stage to the "up" stage.

## Customization

- You can modify the angle thresholds for detecting the stages or customize the drawing styles for the annotations.
