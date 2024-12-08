

https://github.com/user-attachments/assets/463c65a1-29e6-4165-a82e-d840d9f66e52

# Drowsiness Detection System

This project is a real-time drowsiness detection system using computer vision and machine learning. The system leverages OpenCV, Haar cascades for facial and eye detection, and a YOLO model to classify the state of the eyes (open or closed) to determine drowsiness. If the eyes are detected as closed for a prolonged period, an alarm sound is triggered.

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Data Logging](#data-logging)
- [Customization](#customization)
- [License](#license)

---

### Project Structure

- `main.py`: The main script for running the detection system.
- `models/best.pt`: Pre-trained YOLO model file for eye state detection.
- `alarm.wav`: Sound file to alert the user when drowsiness is detected.
- `analysis.csv`: CSV file to log detection results with timestamps and states.

---

### Requirements

- Python 3.x
- OpenCV
- Pygame
- Ultralyics YOLO

### Installation

1. Clone the repository.
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required packages:
    ```bash
    pip install opencv-python pygame ultralytics
    ```

3. Ensure that the pre-trained YOLO model file (`best.pt`) is located in the `models` folder. 

4. Place `alarm.wav` in the main project directory.

---

### Usage

1. Run the main script:
    ```bash
    python main.py
    ```

2. Press `q` to exit the detection system at any time.

---

### How It Works

1. **Face and Eye Detection**: The system uses Haar cascades to detect the face and eyes in each video frame.
  
2. **Eye State Classification**: Every 8 frames, it crops the eye regions and passes them to a YOLO model that classifies the state as "Open" or "Closed."

3. **Score and Alarm Trigger**: If eyes are closed for an extended period (detected by a high score), the system triggers an alarm sound (`alarm.wav`) to alert the user. The score resets when the eyes are detected as open.

4. **Logging**: Each detection result, along with a timestamp, is saved to `analysis.csv`.

### Data Logging

The system records the detection results in `analysis.csv` with the following columns:
- **ID**: Unique identifier for each detection.
- **Timestamp**: Time of the detection.
- **State**: "Open" or "Closed" state of the eyes.

---

### Customization

- **Detection Frequency**: You can adjust `detection_frequency` (set to 8) to run YOLO prediction every X frames for efficiency.
- **Score Threshold**: Modify the `score` threshold (set to 30) to change the duration needed to trigger the alarm.

---

### License

This project is licensed under the MIT License.
