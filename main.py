import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
from tracker import Tracker
import random
from PIL import Image, ImageTk
from collections import defaultdict
import csv
import os


class ObjectDetectionApp:
    """
    A Tkinter application for object detection and tracking using YOLO and Deep SORT.

    Attributes:
    -----------
    root : tk.Tk
        The root window of the Tkinter application.
    frame : tk.Frame
        The frame containing the canvas and buttons.
    canvas : tk.Canvas
        The canvas for displaying video frames and fences.
    select_button : tk.Button
        Button to select a video file.
    process_button : tk.Button
        Button to process the selected video.
    quit_button : tk.Button
        Button to quit the application.
    model : YOLO
        The YOLO model for object detection.
    tracker : Tracker
        The modified DEEP Sort tracker for object tracking.
    colors : list
        List of random colors for drawing bounding boxes.
    video_path : str
        Path to the selected video file.
    class_labels : dict
        Mapping of class IDs to class labels.
    objects : defaultdict
        Dictionary to store detected objects and their data.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection and Tracking")

        # Frame for the canvas and buttons
        self.frame = tk.Frame(root)
        self.frame.pack()

        # Canvas to display video frames and fences
        self.canvas = tk.Canvas(self.frame, width=800, height=600)
        self.canvas.grid(row=0, column=0, columnspan=3)

        # Buttons for video selection, processing, and quitting
        self.select_button = tk.Button(self.frame, text="Select Video", command=self.select_video)
        self.select_button.grid(row=1, column=0, pady=20, padx=20)

        self.process_button = tk.Button(self.frame, text="Process Video", command=self.process_video)
        self.process_button.grid(row=1, column=1, pady=20, padx=20)

        self.quit_button = tk.Button(self.frame, text="Quit", command=root.quit)
        self.quit_button.grid(row=1, column=2, pady=20, padx=20)

        # Initialize YOLO model and tracker
        # self.model = YOLO(model="yolov8m.pt")  # base model
        self.model = YOLO(model="model_data/best.pt")  # custom model
        self.tracker = Tracker()
        self.colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]
        self.videos = None

        # Mapping class IDs to class labels
        # self.class_labels = open("coco.names").read().strip().split('\n')  # base labels
        self.class_labels = open("class.names").read().strip().split('\n')  # custom labels

        # Save each detected object to a dictionary
        self.objects = defaultdict(dict)

    def select_video(self):
        """Select a video file and display its thumbnail."""
        self.video_idx = 0
        self.videos = filedialog.askopenfilenames(title="Select Video File", filetypes=[("Video files", "*.mp4;*.avi")])
        if self.videos:
            self.display_video_thumbnail(self.videos[0])

    def display_video_thumbnail(self, video_path):
        """Display a thumbnail of the selected video on the canvas."""
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (800, 600))
            self.thumbnail = frame
            self.show_frame_on_canvas(frame)
            self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    def show_frame_on_canvas(self, frame):
        """Display a frame on the Tkinter canvas."""
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk

    def process_video(self):
        """Process the selected video and perform object detection and tracking."""
        if not self.videos:
            messagebox.showerror("Error", "No videos selected")
            return
        for video in self.videos:
            self.run_detection(video)
        self.videos = None

    def run_detection(self, video_path):
        """Run YOLO detection and Deep SORT tracking on the video."""
        # Clear any previous object information
        self.objects.clear()

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        frame_idx = 1

        while ret:
            results = self.model(frame, conf=0.5, iou=0.3)
            for result in results:
                detections = [
                    [int(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[5]), r[4]]
                    for r in result.boxes.data.tolist()
                ]

                self.tracker.update(frame, detections)

                for track in self.tracker.tracks:
                    bbox = track.bbox
                    x1, y1, x2, y2 = map(int, bbox)
                    track_id = track.track_id
                    class_id = track.class_id
                    class_label = self.class_labels[class_id]
                    center = track.center
                    confidence = track.confidence

                    print("Track ID:", track_id, "Class ID:", class_id, confidence, "%", class_label, center)

                    coordinates = self.objects[track_id].get('coordinates', []) + [{frame_idx: center}]

                    crossing_payload = {
                        'track_id': track_id,
                        'object_type': class_label,
                        'coordinates': coordinates
                    }

                    self.objects[track_id] = crossing_payload

                    label = f'ID: {track_id} {class_label} {confidence:.2f}%'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors[track_id % len(self.colors)], 3)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors[track_id % len(self.colors)], 3)
                    cv2.circle(frame, center, 5, self.colors[track_id % len(self.colors)], -1)

            frame = cv2.resize(frame, (800, 600))
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            ret, frame = cap.read()
            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()
        self.save_to_csv()
        self.video_idx += 1

    def save_to_csv(self):
        """Save detected object data to a CSV file."""
        title = os.path.splitext(os.path.basename(self.videos[self.video_idx]))[0]
        keys = ['track_id', 'object_type', 'coordinates']
        os.makedirs('./output', exist_ok=True)
        with open(f"./output/{title}.csv", 'w', newline='') as out:
            dict_writer = csv.DictWriter(out, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.objects.values())

        """Save detected object data to a formatted CSV file."""
        title = os.path.splitext(os.path.basename(self.videos[self.video_idx]))[0] + "FORMAT"
        keys = ['track_id', 'object_type', 'frame', 'x', 'y']
        os.makedirs('./output', exist_ok=True)
        with open(f"./output/{title}.csv", 'w', newline='') as out:
            dict_writer = csv.DictWriter(out, fieldnames=keys)
            dict_writer.writeheader()
            rows = []
            for obj in self.objects.values():
                track_id = obj['track_id']
                object_type = obj['object_type']
                for coord in obj['coordinates']:
                    for frame, (x, y) in coord.items():
                        row = {
                            'track_id': track_id,
                            'object_type': object_type,
                            'frame': frame,
                            'x': x,
                            'y': self.video_height - 1 - y  # Flip the y value
                        }
                        rows.append(row)
            dict_writer.writerows(rows)
        

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
