import cv2
import csv
import numpy as np
from .tracker import EuclideanDistTracker

class VehicleCounter():
    def __init__(self, file, video_dim,fps, lineDim, threshold, showVideo, theox, theoy, thedx, thedy):
        # Initialize vehicle counter with the necessary parameters
        # print(' in vehicle counter INIT')
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
        self.oxcoord = theox
        self.oycoord = theoy
        self.dxcoord = thedx
        self.dycoord = thedy
        self.ox2coord = 389
        self.oy2coord = 140
        self.dx2coord = 539
        self.dy2coord = 206
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
            
            # Draw the crossing lines? TODO: Benefits of using this?
            # cv2.line(img, (0, self.middle_line_position), (iw, self.middle_line_position), (255, 0, 255), 2)
            # cv2.line(img, (0, self.middle_line_position), (iw, self.middle_line_position), (255, 0, 255), 2)
            # cv2.line(img, (0, self.up_line_position), (iw, self.up_line_position), (0, 0, 255), 2)
            # cv2.line(img, (0, self.down_line_position), (iw, self.down_line_position), (0, 0, 255), 2)

            # Show the frames
            cv2.imshow('Output', img)
            if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit program
                break

        # Save data to a csv
        with open("./output/data.csv", 'w') as f1:
            cwriter = csv.writer(f1)
            cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
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
        # Draw detection lines and rectangles
        cv2.line(img, (self.oxcoord,self.oycoord), (self.dxcoord,self.dycoord), (187, 0, 255), 2)
        cv2.line(img, (self.oxcoord,self.oycoord), (self.dxcoord,self.oycoord), (187, 0, 255), 2)
        cv2.line(img, (self.dxcoord,self.oycoord), (self.dxcoord,self.dycoord), (187, 0, 255), 2)
        cv2.line(img, (self.dxcoord,self.dycoord), (self.oxcoord,self.dycoord), (187, 0, 255), 2)
        cv2.line(img, (self.oxcoord,self.dycoord), (self.oxcoord,self.oycoord), (187, 0, 255), 2)

        cv2.line(img, (self.ox2coord,self.oycoord), (self.dx2coord,self.dycoord), (187, 0, 255), 2)
        cv2.line(img, (self.ox2coord,self.oycoord), (self.dx2coord,self.oycoord), (187, 0, 255), 2)
        cv2.line(img, (self.dx2coord,self.oycoord), (self.dx2coord,self.dycoord), (187, 0, 255), 2)
        cv2.line(img, (self.dx2coord,self.dycoord), (self.ox2coord,self.dycoord), (187, 0, 255), 2)
        cv2.line(img, (self.ox2coord,self.dycoord), (self.ox2coord,self.oycoord), (187, 0, 255), 2)

    def draw_counting_texts(self, img, font_color, font_size, font_thickness):
        # Draw counting texts in the frame
        cv2.putText(img, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Up2", (210, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Down2", (260, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        
        cv2.putText(img, "Car:        "+str(self.up_list[0])+"     "+ str(self.down_list[0])+"     "+ str(self.up_list[0])+"     "+ str(self.down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Motorbike:  "+str(self.up_list[1])+"     "+ str(self.down_list[1])+"     "+ str(self.up_list[1])+"     "+ str(self.down_list[1]) , (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Bus:        "+str(self.up_list[2])+"     "+ str(self.down_list[2])+"     "+ str(self.up_list[2])+"     "+ str(self.down_list[2]) , (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Truck:      "+str(self.up_list[3])+"     "+ str(self.down_list[3])+"     "+ str(self.up_list[3])+"     "+ str(self.down_list[3]) , (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

    def count_vehicle(self, box_id, img):
        # Count vehicles as they cross the detection lines
        x, y, w, h, id, index = box_id
        # Find the center of the rectangle for detection
        ix, iy = self.find_center(x, y, w, h)

        slope = ( self.dycoord - self.oycoord ) / ( self.dxcoord - self.oxcoord)
        liney = ( slope * ix) - (slope * self.oxcoord) + self.oycoord
                
        # Check if the center is in the top triangle
        if (liney > iy) and (ix > self.oxcoord) and (ix < self.dxcoord) and (iy > self.oycoord) and (iy < self.dycoord):
           print('top triangle here', 'id is ', id,' iy = ', iy , 'liney = ', liney, ' ox = ', self.oxcoord, 'dx = ', self.dxcoord)         
           if id not in self.temp_up_list:
                self.temp_up_list.append(id)
                crossing_info = {
                    'entry_time': 'NOW',  # TODO: Figure out a solution for timestamp. Extrapolate or use relative?
                    'fence_id': 'fence_1',
                    'object_id': id,
                    'object_type': index,
                    'direction': 'up'
                }
                self.crossing_data.append(crossing_info)
                print('added to up list')

        # Check if the center is in the bottom triangle
        elif (liney < iy) and (ix > self.oxcoord) and (ix < self.dxcoord) and (iy > self.oycoord) and (iy < self.dycoord): 
            print('bottom triangle here', 'id is ', id,' iy = ', iy , 'liney = ', liney, ' ox = ', self.oxcoord, 'dx = ', self.dxcoord)
            if id not in self.temp_down_list:
                self.temp_down_list.append(id)
                crossing_info = {
                    'entry_time': 'NOW',  # TODO: Figure out a solution for timestamp. Extrapolate or use relative?
                    'fence_id': 'fence_1',
                    'object_id': id,
                    'object_type': index,
                    'direction': 'down'
                }
                self.crossing_data.append(crossing_info)
                print('added to down list')

        # Check if the vehicle has crossed from bottom to top
        elif (liney > iy) and ((iy < self.oycoord) or (ix > self.dxcoord)):
            #print("above")
            if id in self.temp_down_list:
                print("an upper")
                self.temp_down_list.remove(id)
                self.up_list[index] += 1

        # Check if the vehicle has crossed from top to bottom
        elif (liney < iy) and ((ix < self.oxcoord) or (iy > self.dycoord)):
            #print("below")
            if id in self.temp_up_list:
                print("an downer")
                self.temp_up_list.remove(id)
                self.down_list[index] += 1

              

        # Find the current position of the vehicle
        #    car is below upline and car is above middleline
        #if (iy > self.up_line_position) and (iy < self.middle_line_position):

        #   if id not in self.temp_up_list:
        #        self.temp_up_list.append(id)
        #   car is above down line and car is below middle line
        
        #elif iy < self.down_line_position and iy > self.middle_line_position:
        #   if id not in self.temp_down_list:
        #       self.temp_down_list.append(id)

        #elif iy < self.up_line_position:
        #    if id in self.temp_down_list:
        #        self.temp_down_list.remove(id)
        #        self.up_list[index] = self.up_list[index]+1

        #elif iy > self.down_line_position:
        #    if id in self.temp_up_list:
        #        self.down_list[index] = self.down_list[index] + 1

        # New way is to find the cross product, turn line and point into two vectors
        
        #y1 = self.lineDim[0][1]
        #x2 =  self.lineDim[1][0]
        #y2 = self.lineDim[1][1]
        #xA = ix
        #yA = iy
        #v1 = (x1 - x2,y2-y1)
        #v2 = (x2-xA , y2-yA )
        #xp = v1[0] * v2[1] - v1[1] * v2[0]
        #if(xp > 0):
        #    print('left of line')
        # if(xp < 0):
        #     print('right of line')
        # else:
        #     print('on the line')

        # Draw circle in the middle of the rectangle
        cv2.circle(img, (ix, iy), 2, (0, 0, 255), -1)
        














































# Initialize the videocapture object
# cap = cv2.VideoCapture('C:/Program Files (x86)/Projects/CADSR/VehicleDetection/raw_video/movie2corel.mp4')
# input_size = 320

# Detection confidence threshold
# confThreshold =0.2
# nmsThreshold= 0.2

# font_color = (0, 0, 255)
# font_size = 0.5
# font_thickness = 2

# Middle cross line position
# middle_line_position = 125   
# up_line_position = middle_line_position - 15
# down_line_position = middle_line_position + 15


# Store Coco Names in a list
# classesFile = "C:/Program Files (x86)/Projects/CADSR/VehicleDetection/model/coco.names"
# classNames = open(classesFile).read().strip().split('\n')
# print(classNames)
# print(len(classNames))

# class index for our required detection classes
# required_class_index = [2, 3, 5, 7]

# detected_classNames = []

# ## Model Files
# modelConfiguration = 'C:/Program Files (x86)/Projects/CADSR/VehicleDetection/model/yolov3-320.cfg'
# modelWeigheights = 'C:/Program Files (x86)/Projects/CADSR/VehicleDetection/model/yolov3-320.weights'

# # configure the network model
# net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# # Configure the network backend

# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
# np.random.seed(42)
# colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Function for finding the center of a rectangle
# def find_center(x, y, w, h):
#     x1=int(w/2)
#     y1=int(h/2)
#     cx = x+x1
#     cy=y+y1
#     return cx, cy
    
# List for store vehicle count information
# temp_up_list = []
# temp_down_list = []
# up_list = [0, 0, 0, 0]
# down_list = [0, 0, 0, 0]

# # Function for count vehicle
# def count_vehicle(box_id, img):

#     x, y, w, h, id, index = box_id

#     # Find the center of the rectangle for detection
#     center = find_center(x, y, w, h)
#     ix, iy = center
    
#     # Find the current position of the vehicle
#     if (iy > up_line_position) and (iy < middle_line_position):

#         if id not in temp_up_list:
#             temp_up_list.append(id)

#     elif iy < down_line_position and iy > middle_line_position:
#         if id not in temp_down_list:
#             temp_down_list.append(id)
            
#     elif iy < up_line_position:
#         if id in temp_down_list:
#             temp_down_list.remove(id)
#             up_list[index] = up_list[index]+1

#     elif iy > down_line_position:
#         if id in temp_up_list:
#             temp_up_list.remove(id)
#             down_list[index] = down_list[index] + 1

#     # Draw circle in the middle of the rectangle
#     cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here




# def realTime():
#     while True:
#         success, img = cap.read()
#         print(cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_WIDTH) )

#         img = cv2.resize(img,(0,0),None,.5,.5)
        
#         ih, iw, channels = img.shape
#         blob = cv2.dnn.blobFromImage(img, 1 / 255, (320,320), [0, 0, 0], 1, crop=False)

#         # Set the input of the network
#         net.setInput(blob)
#         layersNames = net.getLayerNames()
#         outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
#         # Feed data to the network
#         outputs = net.forward(outputNames)
    
#         # Find the objects from the network output
#         postProcess(outputs,img)


#         # Draw the crossing lines
#         cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
#         cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 2)
#         cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 2)

#         # Draw counting texts in the frame
#         cv2.putText(img, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
#         cv2.putText(img, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
#         cv2.putText(img, "Car:        "+str(up_list[0])+"     "+ str(down_list[0]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
#         cv2.putText(img, "Motorbike:  "+str(up_list[1])+"     "+ str(down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
#         cv2.putText(img, "Bus:        "+str(up_list[2])+"     "+ str(down_list[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
#         cv2.putText(img, "Truck:      "+str(up_list[3])+"     "+ str(down_list[3]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

#         # Show the frames
#         cv2.imshow('Output', img)

#         if cv2.waitKey(1) == ord('q'):
#             break

#     # Write the vehicle counting information in a file and save it

#     with open("data.csv", 'w') as f1:
#         cwriter = csv.writer(f1)
#         cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
#         up_list.insert(0, "Up")
#         down_list.insert(0, "Down")
#         cwriter.writerow(up_list)
#         cwriter.writerow(down_list)
#     f1.close()
#     # print("Data saved at 'data.csv'")
#     # Finally realese the capture object and destroy all active windows
#     cap.release()
#     cv2.destroyAllWindows()

# image_file = 'vehicle classification-image02.png'

# def from_static_image(image):
#     img = cv2.imread(image)
#     blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_x, input_y), [0, 0, 0], 1, crop=False)
#     # Set the input of the network
#     net.setInput(blob)
#     layersNames = net.getLayerNames()
#     outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
#     # Feed data to the network
#     outputs = net.forward(outputNames)

#     # Find the objects from the network output
#     postProcess(outputs,img)

#     # count the frequency of detected classes
#     frequency = collections.Counter(detected_classNames)
#     print(frequency)
#     # Draw counting texts in the frame
#     cv2.putText(img, "Car:        "+str(frequency['car']), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
#     cv2.putText(img, "Motorbike:  "+str(frequency['motorbike']), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
#     cv2.putText(img, "Bus:        "+str(frequency['bus']), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
#     cv2.putText(img, "Truck:      "+str(frequency['truck']), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)


#     cv2.imshow("image", img)

#     cv2.waitKey(0)

#     # save the data to a csv file
#     with open("static-data.csv", 'a') as f1:
#         cwriter = csv.writer(f1)
#         cwriter.writerow([image, frequency['car'], frequency['motorbike'], frequency['bus'], frequency['truck']])
#     f1.close()


# if __name__ == '__main__':
#     realTime()
#     # from_static_image(image_file)
