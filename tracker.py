from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np

class Tracker:
    """
    A class to perform object tracking using Deep SORT and YOLO for object detection.
    """
    tracker = None
    encoder = None
    tracks = None

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
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
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
            class_ids (list): A list of class IDs corresponding to the tracked objects.
        """
        tracks = []
        for track_idx, track in enumerate(self.tracker.tracks):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            track_id = track.track_id
            class_id = detections[track_idx].class_id if track_idx < len(detections) else -1
            confidence = detections[track_idx].confidence if track_idx < len(detections) else 0
            new_track = Track(track_id, bbox, class_id, confidence)
            tracks.append(new_track)

        self.tracks = tracks

class Track:
    """
    A class to represent a single tracked object.
    """
    track_id = None
    bbox = None
    class_id = None
    center = None
    confidence = None

    def __init__(self, track_id, bbox, class_id, confidence):
        """
        Initializes a Track object with the given parameters.

        Args:
            track_id (int): The unique ID of the track.
            bbox (list): The bounding box coordinates of the tracked object.
            class_id (int): The class ID of the tracked object.
        """
        self.track_id = track_id
        self.bbox = bbox
        self.class_id = class_id
        self.center = self.update_center()
        self.confidence = round(confidence * 100, 2)

    def update_center(self):
        """
        Updates the center point of the bounding box.
        """
        x1, y1, x2, y2 = self.bbox
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        return (center_x, center_y)
