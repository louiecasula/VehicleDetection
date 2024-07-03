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
                    detections.append([x1, y1, x2, y2, score])
                
                self.tracker.update(frame, detections)

                for track in self.tracker.tracks:
                    bbox = track.bbox
                    x1, y1, x2, y2 = bbox
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)
                    track_id = track.track_id

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (self.colors[track_id % len(self.colors)]), 3)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors[track_id % len(self.colors)], 3)

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



# import tkinter as tk
# from tkinter import filedialog, messagebox
# import cv2
# import numpy as np
# from PIL import Image, ImageTk
# from scripts.yolo_detector import YOLODetector
# from scripts.deep_sort import DeepSortTracker

# class App:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Object Detection and Tracking")

#         # Frame for the canvas and buttons
#         self.frame = tk.Frame(root)
#         self.frame.pack()

#         # Canvas to display video frames and fences
#         self.canvas = tk.Canvas(self.frame, width=800, height=600)
#         self.canvas.grid(row=0, column=0, columnspan=3)

#         # Buttons for video selection, processing, and quitting
#         self.select_button = tk.Button(self.frame, text="Select Video", command=self.select_video)
#         self.select_button.grid(row=1, column=0, pady=20, padx=20)

#         self.process_button = tk.Button(self.frame, text="Process Video", command=self.process_video)
#         self.process_button.grid(row=1, column=1, pady=20, padx=20)

#         self.quit_button = tk.Button(self.frame, text="Quit", command=root.quit)
#         self.quit_button.grid(row=1, column=2, pady=20, padx=20)

#         self.fence = None
#         self.drawing = False
#         self.start_x, self.start_y = None, None
#         self.end_x, self.end_y = None, None
#         self.canvas.bind("<ButtonPress-1>", self.on_button_press)
#         self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
#         self.canvas.bind("<Motion>", self.on_mouse_move)
#         self.yolo = YOLODetector()
#         self.deepsort = DeepSortTracker()
#         self.video_path = None

#     def select_video(self):
#         self.video_path = filedialog.askopenfilename()
#         self.display_video_thumbnail(self.video_path)

#     def display_video_thumbnail(self, video_path):
#         cap = cv2.VideoCapture(video_path)
#         ret, frame = cap.read()
#         if ret:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = cv2.resize(frame, (800, 600))
#             self.thumbnail = frame
#             self.show_frame_on_canvas(frame)
#         cap.release()

#     def show_frame_on_canvas(self, frame):
#         img = Image.fromarray(frame)
#         imgtk = ImageTk.PhotoImage(image=img)
#         self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
#         self.canvas.imgtk = imgtk

#     def on_button_press(self, event):
#         self.drawing = True
#         self.start_x, self.start_y = event.x, event.y

#     def on_button_release(self, event):
#         self.drawing = False
#         self.end_x, self.end_y = event.x, event.y
#         self.fence = (self.start_x, self.start_y, self.end_x, self.end_y)
#         self.canvas.create_line(self.start_x, self.start_y, self.end_x, self.end_y, fill="red", width=2)

#     def on_mouse_move(self, event):
#         if self.drawing:
#             self.canvas.delete("temp_line")
#             self.canvas.create_line(self.start_x, self.start_y, event.x, event.y, fill="red", width=2, tags="temp_line")

#     def process_video(self):
#         if self.video_path is None:
#             messagebox.showerror("Error", "No video selected")
#             return
#         if self.fence is None:
#             messagebox.showerror("Error", "No fence drawn")
#             return
#         self.run_detection(self.video_path)

#     def run_detection(self, video_path):
#         cap = cv2.VideoCapture(video_path)
#         frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         output_video_path = video_path.split('.')[0] + '_output.avi'
#         out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Perform detection
#             results = self.yolo.detect(frame)  # Assuming `self.yolo` returns results in the correct format
            
#             # Ensure results are in the expected format
#             if results is None or len(results) == 0:
#                 formatted_detections = []
#             else:
#                 detections = results if isinstance(results, np.ndarray) else np.array(results)
                
#                 # Debug print statements
#                 print("YOLO Detections:", detections)
                
#                 formatted_detections = []
#                 for det in detections:
#                     print("Detection:", det)
#                     x_center, y_center, width, height, confidence, class_id = det[:6]  # Extract the first six elements
#                     formatted_detections.append([x_center, y_center, width, height, confidence, class_id])

#                 # Convert to numpy array
#                 formatted_detections = np.array(formatted_detections)
#                 print("Formatted Detections:", formatted_detections)
            
#             if len(formatted_detections) > 0:
#                 tracks = self.deepsort.update(formatted_detections, frame=frame)
#             else:
#                 tracks = self.deepsort.update([], frame=frame)
            
#             # Draw detections and tracks on the frame
#             for track in tracks:
#                 bbox = track.to_tlbr()  # Get the bounding box in top-left to bottom-right format
#                 track_id = track.track_id  # Get the ID for the track
#                 class_id = int(track.det_class)  # Get the detected class
#                 color = self.colors[class_id % len(self.colors)]  # Choose a color for the bounding box
                
#                 cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
#                 cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            
#             out.write(frame)
        
#         cap.release()
#         out.release()
#         cv2.destroyAllWindows()


#     def crossed_fence(self, bbox, fence):
#         # frame = cv2.resize(frame, (800, 600))
#         x1, y1, x2, y2 = bbox
#         cx = (x1 + x2) // 2
#         cy = (y1 + y2) // 2
#         fx1, fy1, fx2, fy2 = fence
#         cross_product = (fx2 - fx1) * (cy - fy1) - (fy2 - fy1) * (cx - fx1)
#         return cross_product > 0

#     def update_tabulations(self, object_id, class_id, confidence):
#         # Add your code to update the tabulations
#         pass

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = App(root)
#     root.mainloop()
