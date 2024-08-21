from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np


class Tracker:
    """
    A class to perform object tracking using Deep SORT and YOLO for object detection.

    Attributes:
    -----------
    tracker : DeepSortTracker
        The Deep SORT tracker instance.
    encoder : Callable
        The function to generate features for bounding boxes.
    tracks : list
        List of current tracks.
    """

    def __init__(self):
        """
        Initializes the Tracker with necessary configurations and models.
        """
        max_cosine_distance = 0.4
        nn_budget = None
        encoder_model_filename = 'model_data/mars-small128.pb'

        # Create the nearest neighbor distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # Initialize the Deep SORT tracker
        self.tracker = DeepSortTracker(metric)
        # Create the box encoder using the specified model
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)
        self.tracks = []

    def update(self, frame, detections):
        """
        Updates the tracker with the new frame and detections.

        Args:
            frame (numpy.ndarray): The current frame from the video.
            detections (list): A list of detections, where each detection is a list containing
                               bounding box coordinates, class ID, and score.
        """
        if len(detections) == 0:
            # If no detections, only predict and update with no detections
            self.tracker.predict()
            self.tracker.update([])  
            self.update_tracks([])
            return

        # Extract bounding boxes and class IDs from detections
        bboxes = np.asarray([d[:-2] for d in detections])  # Exclude class_id and score
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]  # Convert to (x, y, width, height)
        scores = [d[-1] for d in detections]
        class_ids = [d[-2] for d in detections]

        # Generate features for the detections
        features = self.encoder(frame, bboxes)

        # Create Detection objects for the tracker
        dets = [Detection(bbox, scores[i], features[i], class_ids[i]) for i, bbox in enumerate(bboxes)]

        # Predict and update the tracker with new detections
        self.tracker.predict()
        self.tracker.update(dets)
        # Update the internal tracks with the detected class IDs
        self.update_tracks(dets)

    def update_tracks(self, detections):
        """
        Updates the internal tracks with new class IDs.

        Args:
            detections (list): A list of Detection objects.
        """
        tracks = []
        i = 0
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            track.class_id = detections[i].class_id
            track.confidence = round(detections[i].confidence * 100, 2)
            track.bbox = track.to_tlbr()
            track.center = track.update_center()
            tracks.append(track)
            i += 1

        self.tracks = tracks

