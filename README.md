# Real-Time Object Detection and Tracking Project


This project utilizes the Ultralytics YOLO model to perform real-time object detection and tracking, focusing on specific classes such as cars and persons. It uniquely visualizes bounding boxes, scores, and IDs on detected objects while also evaluating system performance through comprehensive metrics. The modular coding approach enhances extendibility and reusability.

## Features

- **Dynamic Object Tracking:** Targets specific objects based on `class_id` (e.g., humans or cars).
- **Visualization:** Visualizes bounding boxes, scores, and IDs on detected objects in real-time.
- **Performance Analysis:** Evaluates system performance using metrics like FPS and average processing time per frame.
- **Modular Design:** Facilitates easy adaptation and extension through modular and reusable code components.

## Demo Videos

Check out our demonstration videos showing the project's capability in tracking cars and persons. These videos are stored in the `assets` folder:

- **Car Tracking:** `assets/car.mp4`
- **Person Tracking:** `assets/person.mp4`

Note: Due to GitHub's limitations, direct video playback in the README is not supported. Please download the videos from the `assets` folder to view them.

## Getting Started

Follow these simple steps to get a local copy up and running.

### Prerequisites

- Python 3.6 or later
- OpenCV library
- Ultralytics YOLO model

### Installation - Useage

* Clone the repository:
   ```sh
   git clone https://github.com/yourusername/yourprojectname.git

* To run the main script and start processing your video feed:
   ```sh
   python main.py
