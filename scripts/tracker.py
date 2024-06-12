import math

class EuclideanDistTracker:
    def __init__(self):
        """
        Initializes the tracker with an empty dictionary to store the center positions of objects and 
        an ID count that increments for each new object detected.
        """
        self.center_points = {}  # Dictionary to store the center positions of the objects
        self.id_count = 0  # Counter to assign unique IDs to new objects


    def update(self, objects_rect: list) -> list:
        """
        Updates the tracker with the current frame's detected object bounding boxes.

        Args:
            objects_rect (list): A list of bounding boxes for detected objects in the format [x, y, w, h, index].
                x, y - top-left coordinates of the bounding box
                w, h - width and height of the bounding box
                index - additional index or label associated with the object

        Returns:
            list: A list of bounding boxes with their associated IDs in the format [x, y, w, h, id, index].
        """
        objects_bbs_ids = []  # List to store bounding boxes and their associated IDs

        for rect in objects_rect:
            x, y, w, h, index = rect
            cx = (x + x + w) // 2  # Calculate the center x-coordinate of the bounding box
            cy = (y + y + h) // 2  # Calculate the center y-coordinate of the bounding box

            # Check if the object was detected in the previous frame
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])  # Calculate Euclidean distance between current and previous center points

                if dist < 200:  # If the distance is less than the threshold, consider it the same object
                    self.center_points[id] = (cx, cy)  # Update the center point for this object ID
                    objects_bbs_ids.append([x, y, w, h, id, index])
                    same_object_detected = True
                    break

            # If it's a new object, assign a new ID
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count, index])
                self.id_count += 1

        # Clean up the dictionary to remove IDs that are no longer in use
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id, index = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()  # Update the dictionary with active object IDs
        return objects_bbs_ids  # Return the list of bounding boxes with IDs
