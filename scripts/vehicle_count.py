import cv2
import csv
import numpy as np
import datetime
from .tracker import EuclideanDistTracker

vehicles = ['car', 'motorbike', 'bus', 'truck']

class VehicleCounter():
    def __init__(self, file, video_dim, fps, lineDim, threshold, showVideo, oxcoord, oycoord, dxcoord, dycoord):
        # Initialize vehicle counter with the necessary parameters
        self.file = file
        self.video_dim = video_dim
        self.fps = fps
        self.lineDim = lineDim
        self.showVideo = showVideo
        self.tracker = EuclideanDistTracker()
        self.middle_line_position = 155
        self.up_line_position = 155 - threshold
        self.down_line_position = 155 + threshold
        self.required_class_names = []
        self.required_class_index = [2, 3, 5, 7]
        self.temp_up_list = []
        self.temp_down_list = []
        self.temp_up_list2 = []
        self.temp_down_list2 = []
        self.detected_classNames = []
        self.up_list = [0, 0, 0, 0]
        self.down_list = [0, 0, 0, 0]
        self.up_list2 = [0, 0, 0, 0]
        self.down_list2 = [0, 0, 0, 0]
        self.oxcoord = oxcoord
        self.oycoord = oycoord
        self.dxcoord = dxcoord
        self.dycoord = dycoord
        self.crossing_data = []

    def find_center(self, x, y, w, h):
        # Calculate the center of a bounding box
        return (x + w // 2, y + h // 2)

    def postProcess(self,outputs,img):
        # Process the output from the neural network to find detected objects
        np.random.seed(42)
        confThreshold = 0.2
        nmsThreshold = 0.2
        classesFile = "./model/coco.names"
        classNames = open(classesFile).read().strip().split('\n')
        colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')
        height, width = img.shape[:2]
        boxes = []
        classIds = []
        confidence_scores = []
        detection = []
        
        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if classId in self.required_class_index:
                    if confidence > confThreshold:
                        w,h = int(det[2]*width) , int(det[3]*height)
                        x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                        boxes.append([x,y,w,h])
                        classIds.append(classId)
                        confidence_scores.append(float(confidence))

        # Apply Non-Max Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
        
        if len(indices) > 0:
            try:
                indices.flatten()
            except NameError:
                pass
            else:
                for i in indices.flatten():
                    x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                    color = [int(c) for c in colors[classIds[i]]]
                    name = classNames[classIds[i]]
                    self.detected_classNames.append(name)
                    # Draw classname and confidence score 
                    cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    # Draw bounding rectangle
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                    detection.append([x, y, w, h, self.required_class_index.index(classIds[i])])

        # Update the tracker for each object
        boxes_ids = self.tracker.update(detection)
        for box_id in boxes_ids:
                self.count_vehicle(box_id, img)

    def start(self):
        # Start video capture and processing
        cap = cv2.VideoCapture(self.file)
        font_color = (0, 0, 255)
        font_size = 0.5
        font_thickness = 2
        # Model Files
        modelConfiguration = './model/yolov3-320.cfg'
        modelWeigheights = './model/yolov3-320.weights'
        
        # Configure the network model
        net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)        
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Resize video for processing
        while True:
            success, img = cap.read()
            img = cv2.resize(img,(0,0),None,.5,.5)  # TODO: pass video resize options as parameter in GUI
            ih, iw, channels = img.shape
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (320,320), [0, 0, 0], 1, crop=False)
            # Set the input of the network
            net.setInput(blob)
            layersNames = net.getLayerNames()
            outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
            # Feed data to the network
            outputs = net.forward(outputNames)
            # Find the objects from the network output
            self.postProcess(outputs,img)

            # Draw detection lines and rectangles
            self.draw_lines(img)

            # Draw counting texts in the frame
            self.draw_counting_texts(img, font_color, font_size, font_thickness)

            # Show the frames
            cv2.imshow('Output', img)
            if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit program
                break

        # Save data to a csv
        with open("./output/data.csv", 'w') as f1:
            cwriter = csv.writer(f1)
            cwriter.writerow(['Direction'] + vehicles)
            self.up_list.insert(0, "Up")
            self.down_list.insert(0, "Down")
            cwriter.writerow(self.up_list)
            cwriter.writerow(self.down_list)
        f1.close()
        print("Data saved at 'data.csv'")

        # Save data to a csv RACCA style
        keys = ['entry_time', 'fence_id', 'object_id', 'object_type', 'direction']
        with open("./output/data1.csv", 'w', newline='') as out:
            dict_writer = csv.DictWriter(out, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.crossing_data)
        out.close()
        print("Data saved at 'data1.csv'")

        # Release the capture object and destroy all active windows
        cap.release()
        cv2.destroyAllWindows()

    def draw_lines(self, img):
        # Draw detection line
        cv2.line(img, (self.oxcoord,self.oycoord), (self.dxcoord,self.dycoord), (187, 0, 255), 2)

    def draw_counting_texts(self, img, font_color, font_size, font_thickness):
        # Draw counting texts in the frame
        cv2.putText(img, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        
        cv2.putText(img, "Car:        "+str(self.up_list[0])+"     "+ str(self.down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Motorbike:  "+str(self.up_list[1])+"     "+ str(self.down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Bus:        "+str(self.up_list[2])+"     "+ str(self.down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Truck:      "+str(self.up_list[3])+"     "+ str(self.down_list[3]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

    def count_vehicle(self, box_id, img):
        # Count vehicles as they cross the detection lines
        x, y, w, h, id, index = box_id
        # Find the center of the rectangle for detection
        ix, iy = self.find_center(x, y, w, h)

        # Calculate the position of the vehicle relative to the line
        line_vector = (self.dxcoord - self.oxcoord, self.dycoord - self.oycoord)
        point_vector = (ix - self.oxcoord, iy - self.oycoord)
        cross_product = line_vector[0] * point_vector[1] - line_vector[1] * point_vector[0]
                
        # Check if the center is in the top triangle
        direction = ""
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        if cross_product > 0:
            # Vehicle is on one side of the line
            if id not in self.temp_up_list:
                self.temp_up_list.append(id)
                direction = "up"
        elif cross_product < 0:
            # Vehicle is on the other side of the line
            if id not in self.temp_down_list:
                self.temp_down_list.append(id)
                direction = "down"

        if direction:
            crossing_payload = {
                'entry_time': current_time,  # TODO: Figure out how to grab timestamp from video (frame?)
                'fence_id': 'fence_1',  # TODO: Add field for this once multiple fences can be added.
                'object_id': id,
                'object_type': vehicles[index],
                'direction': direction
            }
            self.crossing_data.append(crossing_payload)
            print('added to up list')

        # Check if the vehicle has crossed from bottom to top
        if id in self.temp_down_list and cross_product > 0:
            self.temp_down_list.remove(id)
            self.up_list[index] += 1
        
        # Check if the vehicle has crossed from top to bottom
        if id in self.temp_up_list and cross_product < 0:
            self.temp_up_list.remove(id)
            self.down_list[index] += 1

        # Draw circle in the middle of the rectangle
        cv2.circle(img, (ix, iy), 2, (0, 0, 255), -1)
        