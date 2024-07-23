# Vehicle Tracking with YOLOv8 and Deep SORT

This project leverages YOLOv8 for object detection and Deep SORT for object tracking to monitor vehicle movements in traffic intersection videos. The application detects various types of vehicles and tracks their trajectories, providing insightful data for traffic analysis.

## Installation

### Prerequisites

- Python 3.8 or higher

### Steps

1. Clone the repository:

    ```sh
    git clone https://github.com/louiecasula/VehicleDetection.git
    ```

2. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Running the Application

1. Ensure you have a video file you wish to process.
2. Execute the application:

    ```sh
    python main.py
    ```

3. The GUI will prompt you to select a video file for processing.

![videointerp_demo1](https://github.com/user-attachments/assets/74a4ef9b-520e-4b1c-b68d-dfd7969f90f9)


### Vehicle Tracking

- The application will detect and track vehicles within the selected video.
- The center points of detected vehicles are used to calculate their direction of travel.
- As each vehicle moves, its center point is tracked, and the direction is determined based on these points' movement across frames.
- Each tracked vehicle's data is saved into a CSV file, capturing:
  - Track ID
  - Class/Object type
  - Coordinates mapped to the corresponding frame

### OpenCV Integration

- **OpenCV** is utilized extensively for various video processing tasks, including video file reading, frame extraction, and displaying the processed video with overlayed tracking information.

## YOLOv8 and Deep SORT

- **YOLOv8**: Employed for real-time object detection, providing bounding boxes and class labels for detected vehicles.
- **Deep SORT**: Utilized for object tracking, assigning unique IDs to detected objects and maintaining their identities across frames, even when temporarily occluded.

### References

- The implementation of YOLOv8 and Deep SORT is based on guidance from this [repository](https://github.com/computervisioneng/object-tracking-yolov8-deep-sort).
- Modifications were made to the Deep SORT implementation from this [repository](https://github.com/nwojke/deep_sort), including updates to the `Detections` object and import statements to ensure compatibility with the current setup.

## Future Considerations

- **Graphical Representation of Vehicle Paths**: Enhancing the application to visually represent the center coordinates of each vehicle, illustrating their traveled paths.
- **Custom Model Training**: Exploring the training of custom models to enhance detection accuracy.
- **Alternative Detectors**: Investigating the use of other object detection models for comparative analysis and potential improvement in tracking performance.
