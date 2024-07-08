import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
from tracker import Tracker
import random
import numpy as np
from PIL import Image, ImageTk

class ObjectDetectionApp:
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
        self.model = YOLO("yolov8n.pt")
        self.tracker = Tracker()
        self.colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]
        self.video_path = None

        # Mapping class IDs to class labels
        self.class_labels = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
            4: "airplane", 5: "bus", 6: "train", 7: "truck",
            8: "boat", 9: "traffic light", 10: "fire hydrant",
        }

    def select_video(self):
        # Select a video file
        self.video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4;*.avi")])
        self.display_video_thumbnail(self.video_path)

    def display_video_thumbnail(self, video_path):
        # Display thumbnail of the selected video
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (800, 600))
            self.thumbnail = frame
            self.show_frame_on_canvas(frame)
        cap.release()

    def show_frame_on_canvas(self, frame):
        # Show the frame on the Tkinter canvas
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk

    def process_video(self):
        # Process the selected video
        if self.video_path is None:
            messagebox.showerror("Error", "No video selected")
            return
        self.run_detection(self.video_path)

    def run_detection(self, video_path):
        # Run YOLO detection and Deep SORT tracking
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()

        while ret:
            results = self.model(frame)
            for result in results:
                detections = []
                for r in result.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    class_id = int(class_id)
                    detections.append([x1, y1, x2, y2, class_id, score])
                
                self.tracker.update(frame, detections)

                for track in self.tracker.tracks:
                    bbox = track.bbox
                    x1, y1, x2, y2 = bbox
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    track_id = track.track_id
                    class_id = track.class_id
                    class_label = self.class_labels.get(class_id, "unknown")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (self.colors[track_id % len(self.colors)]), 3)
                    cv2.putText(frame, f"ID: {track_id} {class_label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors[track_id % len(self.colors)], 3)

            frame = cv2.resize(frame, (800, 600))
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            ret, frame = cap.read()

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()

