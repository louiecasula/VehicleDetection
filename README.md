# OpenCV Vehicle Detection
A deployable model built upon the OpenCV computer vision library for the use of classifying vehicles and counting how many of each enter and exit a particular subdivision over a period of time.

## Features

- **Video Analysis**: Processes video footage to detect various objects (vehicles, pedestrians) at traffic intersections.
- **Graphical User Interface (GUI)**: User-friendly interface for video selection and fence drawing.
- **Object Detection**: Uses OpenCV to identify and track objects in the video.
- **Data Tabulation**: Records detected objects crossing a user-defined fence and stores the data in an Excel spreadsheet.

## Installation

### Prerequisites

- Python 3.8
- Anaconda (recommended for managing dependencies)
- OpenCV
- Tkinter

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/louiecasula/VehicleDetection.git
   cd VehicleDetection
   ```

2. **Create a new conda environment:**
   ```bash
   conda create --name vehicledetect
   conda activate vehicledetect
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install OpenCV:**
   ```bash
   conda install -c conda-forge opencv
   ```

## Usage

1. **Launch the program:**
   ```bash
   python main.py
   ```

2. **Select a video:**
   - The GUI will open, prompting you to select a video file from your machine.

3. **Draw a fence:**
   - After selecting the video, draw a fence on the video frame. The program will count objects crossing this fence.

4. **Start detection:**
   - Click the "Process Video" button. The program will process the video, detect objects, and update the table when objects cross the fence.

5. **Export data:**
   - The data will be automatically tabulated into a spreadsheet located in the output directory.
