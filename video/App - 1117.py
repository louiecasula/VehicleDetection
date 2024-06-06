from email import header
from tkinter import Tk,filedialog,E,END, NW,SW, W, Label,Frame,Entry,Button,StringVar
from PIL import Image,ImageTk
# from scripts.vehicle_count import *
import cv2
import os
from scripts.utils import validateThreshold

from scripts.vehicle_count import VehicleCounter
from math import sqrt, floor

class App(Tk):
    def __init__(self):
        super().__init__()
        self.geometry("1200x700")
        self.title("Vehicle Detection")
        self.header = header
        self.clicked_coords = (0,0)
        self.line_coords = [(0,0),(0,0)]
        self.raw_image = []
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=4)
        self.rowconfigure(1, weight=1)
        self.buildGui()


    def buildGui(self):
        
        def processVideoApply():
            print('In processvideoapply')
            file = input_video_val.get()
            showVideo = True
            video_dim = (200,200)
            fps = 30
            cleanThreshold = validateThreshold(threshold_val.get())
            if(not cleanThreshold):
                return False
            tracker = VehicleCounter(file, video_dim,fps, self.line_coords, 15, showVideo)
            tracker.start()

        def buildThreshold(img, thickness, origin_coords,destination_coords):
            cleanThreshold = validateThreshold(threshold_val.get())
            if(not cleanThreshold):
                return False
            (dx, dy) = (destination_coords[0] - origin_coords[0], destination_coords[1] - origin_coords[1])
            len = sqrt((destination_coords[0] - origin_coords[0])**2 + (destination_coords[1] - origin_coords[1])**2)
            (udx, udy) = (dx / len, dy / len)
            (px, py) = (-udy, udx)
            (x1p, y1p) = (origin_coords[0] + px * cleanThreshold, origin_coords[1] + py * cleanThreshold)  
            (x2p, y2p) = (x1p + dx, y1p + dy)
            (x3p, y3p) = (origin_coords[0] - px * cleanThreshold, origin_coords[1] - py * cleanThreshold)  
            (x4p, y4p) = (x3p + dx, y3p + dy)

            (pt1x, pt1y) = (origin_coords[0],origin_coords[1])
            (pt2x, pt2y) = (origin_coords[0],destination_coords[1])
            (pt3x, pt3y) = (destination_coords[0],destination_coords[1])
            (pt4x, pt4y) = (destination_coords[0],origin_coords[1])

            #edited_imaged = cv2.line(img=img,pt1=(floor(x1p),floor(y1p)),pt2=(floor(x2p),floor(y2p)),color=(0, 255, 0),thickness=thickness)
            #edited_imaged = cv2.line(img=img,pt1=(floor(x3p),floor(y3p)),pt2=(floor(x4p),floor(y4p)),color=(0, 255, 0),thickness=thickness)

            #box corner approach
            edited_imaged = cv2.line(img=img,pt1=(floor(pt1x),floor(pt1y)),pt2=(floor(pt2x),floor(pt2y)),color=(0, 255, 0),thickness=thickness)
            edited_imaged = cv2.line(img=img,pt1=(floor(pt2x),floor(pt2y)),pt2=(floor(pt3x),floor(pt3y)),color=(0, 255, 0),thickness=thickness)
            edited_imaged = cv2.line(img=img,pt1=(floor(pt3x),floor(pt3y)),pt2=(floor(pt4x),floor(pt4y)),color=(0, 255, 0),thickness=thickness)
            edited_imaged = cv2.line(img=img,pt1=(floor(pt4x),floor(pt4y)),pt2=(floor(pt1x),floor(pt1y)),color=(0, 255, 0),thickness=thickness)
            return edited_imaged

        def get_click_coords(event, raw_image):
            edited_imaged = raw_image.copy()
            if(self.clicked_coords == (0,0)):
                origin_coord = f'({event.x} , {event.y})'
                self.line_coords[0] = (event.x,event.y)
                edited_imaged = cv2.putText(img=edited_imaged,text=origin_coord,org=(0,50),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0,0))
                edited_imaged = cv2.putText(img=edited_imaged,text=" -> ", org=(200,50),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 0))
                origin_coord_val.delete(0, END)
                origin_coord_val.insert(0,f'{origin_coord}')
                edited_imaged = Image.fromarray(edited_imaged)
                imgtk = ImageTk.PhotoImage(image=edited_imaged)
                lmain.imgtk = imgtk
                lmain.configure(image=imgtk)
                self.set_coords(event.x,event.y)
            else:
                origin_coord = f'({self.clicked_coords[0]} , {self.clicked_coords[1]})'
                destination_coord = f'({event.x} , {event.y})'
                self.line_coords[1] = (event.x,event.y)
                edited_imaged = cv2.putText(img=edited_imaged,text=origin_coord, org=(0,50),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 0))
                edited_imaged = cv2.putText(img=edited_imaged,text=" -> ", org=(200,50),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 0))
                edited_imaged = cv2.putText(img=edited_imaged,text=destination_coord, org=(300,50),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1, color=(255, 0, 0))
                dest_coord_val.delete(0, END)
                dest_coord_val.insert(0,f'{destination_coord}')
                edited_imaged = cv2.line(img=edited_imaged,pt1=self.clicked_coords,pt2=(event.x,event.y),color=(0, 255, 0),thickness=3)
                edited_imaged = buildThreshold(edited_imaged,1,self.clicked_coords,(event.x,event.y))
                edited_imaged = Image.fromarray(edited_imaged)
                imgtk = ImageTk.PhotoImage(image=edited_imaged)
                lmain.imgtk = imgtk
                lmain.configure(image=imgtk)
                self.set_coords(0,0)


        def video_apply():
            print('in video apply')
            filename = filedialog.askopenfile(
            title='Open a file',
            initialdir='./rawVideo',
            filetypes = (
                ('mp4 files', '*.mp4'),
                ('wav files', '*.wav*'),
                ('avi files','*.avi'),
                ('all files','*')
            )
            )
            input_video_val.insert(0,filename.name.replace('/', '\\'))
            cap = cv2.VideoCapture(filename.name)
            success,image = cap.read()
            if success:
                #image = cv2.resize(image, (500, 380))
                image = cv2.resize(image, (682, 384))  
                self.set_raw_image(image)
                img = Image.fromarray(image)
                imgtk = ImageTk.PhotoImage(image=img)
                lmain.imgtk = imgtk
                lmain.configure(image=imgtk)
                lmain.bind( "<Button>", lambda evt: get_click_coords(evt,self.raw_image) )  
            cap.release()

        # # Create Left side inputs column
        inputs_frame = Frame(self)
        inputs_frame.grid(column=0, row=0,columnspan = 3, rowspan = 3, sticky=NW)

        # # Create Right side video column
        video_frame = Frame(self)
        video_frame.grid(column=1, row=0, sticky=E, columnspan = 1)

        # # Create Footer
        footer = Frame(self)
        footer.grid(column=0, row=4)

        # # Add box for video preview
        lmain = Label(video_frame)
        lmain.grid(column=0,row=0,sticky=NW,padx = 20,pady = 20)


        # # Top level instructions
        instructions = Label(master=inputs_frame,text="Drag an mp4 or .avi onto the bar")
        instructions.grid(column=0,row=0, sticky=NW)

        # # Root directory input
        root_label = Label(inputs_frame, text="Root Directory")
        root_label.grid(row = 1, column = 0,pady = 2, sticky=W)
        root_val = Entry(inputs_frame, bd =5, w="70")
        root_val.insert(0, os.getcwd() )
        root_val.grid(row = 1, column = 1, pady = 2, sticky=W)


        # # File input and apply input
        strVar = StringVar()
        input_video_label = Label(master=inputs_frame, text="Input Video")
        input_video_label.grid(row = 2, column = 0, sticky = W, pady = 20,)
        input_video_val = Entry(master=inputs_frame, textvar=strVar, bd=5,w="70")
        #input_video_val.drop_target_register(DND_FILES)
        apply_video_btn = Button( master=inputs_frame, text=u"\U0001F4C1", command=video_apply )

        apply_video_btn.grid(row = 2, column = 2, sticky = W, pady = 20)
        input_video_val.grid(row = 2, column = 1, sticky =W, pady = 20)

        # # Threshold Input
        threshold_label = Label(master=inputs_frame, text="Threshold")
        threshold_label.grid(row = 4, column = 0, sticky = W, pady = 20,) 
        threshold_val = Entry(master=inputs_frame, textvar="15", bd=5,w="15")
        threshold_val.insert(0, "15")
        threshold_val.grid(row=4,column=1, columnspan=1,sticky=W)

        # # Coords of line
        coords_frame = Frame(inputs_frame)
        spacer = Label(master=coords_frame,text="",)
        spacer.grid(row=0,column=1)
        coords_frame.grid(row=3,column=0, columnspan=4,sticky=W)
        coords_label = Label(master=coords_frame, text="Line Coords")
        coords_label.grid(row=0,column=0, sticky = W)
        origin_coord_val = Entry(master=coords_frame,  bd=5,w="15")
        dest_coord_val = Entry(master=coords_frame, bd=5,w="15")
        origin_coord_val.grid(row=0,column=3,sticky = W)
        dest_coord_val.grid(row=0,column=4, sticky = W)

        # Process button
        process_button = Button ( footer, text ="Process Video", command=processVideoApply)
        process_button.grid(column=0,row=0, sticky=SW)

        

    def set_coords(self,x,y):
        self.clicked_coords = (x,y)


    def set_raw_image(self,imageArr):
        self.raw_image = imageArr
     
    
    
