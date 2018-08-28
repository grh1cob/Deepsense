import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import numpy as np
import os
from . import framework
import csv

class App:
    def __init__(self, window, window_title, video_source, tracker, encoder):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
        elapsed = 0
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 100
        self.update(elapsed)
        self.tracker = tracker
        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


    def update(self, elapsed):
        # Get a frame from the video source
        elapsed += 1
        ret, frame = self.vid.get_frame()
        preprocessed = framework.preprocess(frame)
        # single_out =
        postprocessed = framework.postprocess(preprocessed, frame,frame_id = elapsed,tracker=None)

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(postprocessed))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)

class MyVideoCapture:
    def __init__(self, video_source):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
def gui(self):
    source = self.FLAGS.demo
    SaveVideo = self.FLAGS.saveVideo
    if self.FLAGS.track :
        if self.FLAGS.tracker == "deep_sort":
            from deep_sort import generate_detections
            from deep_sort.deep_sort import nn_matching
            from deep_sort.deep_sort.tracker import Tracker
            metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", 0.2, 100)
            tracker = Tracker(metric)
            encoder = generate_detections.create_box_encoder(
                os.path.abspath("deep_sort/resources/networks/mars-small128.ckpt-68577"))
        elif self.FLAGS.tracker == "sort":
            from sort.sort import Sort
            encoder = None
            tracker = Sort()

    if self.FLAGS.BK_MOG and self.FLAGS.track :
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    if self.FLAGS.csv :
        f = open('{}.csv'.format(file),'w')
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['frame_id', 'track_id' , 'x', 'y', 'w', 'h'])
        f.flush()
    else :
        f =None
        writer= None

    App(tkinter.Tk(), "Tkinter and OpenCV", 0, tracker, encoder)
