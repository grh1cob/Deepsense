from darkflow.darkflow.defaults import argHandler #Import the default arguments
import os
from darkflow.darkflow.net.build import TFNet
from collections import deque
import cv2
import numpy as np
from time import time as timer
import sys
# FLAGS = argHandler()
# FLAGS.setDefaults()

FLAGS = {
    'demo': "darkflow/sample_vid/multiple.mp4",
    'model': 'darkflow/cfg/yolov2-voc-2c.cfg',
    'load': 14300,
    'threshold': 0.4,
    'gpu': 0.8,
    'display': True,
    'track': True,
    'trackObj': ['drone', 'bird'],
    'saveVideo': False,
    'BK_MOG': False,
    'tracker': 'deep_sort',
    'skip': 0,
    'csv': False
}
buff = deque( maxlen=20 )
tfnet = TFNet(FLAGS)
# from piyush import Connect
file = FLAGS["demo"]
SaveVideo = FLAGS["saveVideo"]
# w = Connect()
if FLAGS["track"] :
    if FLAGS["tracker"] == "deep_sort":
        from deep_sort import generate_detections
        from deep_sort.deep_sort import nn_matching
        from deep_sort.deep_sort.tracker import Tracker
        metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", 0.2, 100)
        tracker = Tracker(metric)
        encoder = generate_detections.create_box_encoder(
            os.path.abspath("deep_sort/resources/networks/mars-small128.ckpt-68577"))
    elif FLAGS["tracker"] == "sort":
        from sort.sort import Sort
        encoder = None
        tracker = Sort()
if FLAGS["BK_MOG"] and FLAGS["track"] :
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

if file == 'camera':
    file = 0
else:
    assert os.path.isfile(file), \
    'file {} does not exist'.format(file)

camera = cv2.VideoCapture(file)

assert camera.isOpened(), \
'Cannot capture source'

if FLAGS["csv"] :
    f = open('{}.csv'.format(file),'w')
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['frame_id', 'track_id' , 'x', 'y', 'w', 'h'])
    f.flush()
else :
    f =None
    writer= None
if file == 0:#camera window
    cv2.namedWindow('', 0)
    _, frame = camera.read()
    height, width, _ = frame.shape
    cv2.resizeWindow('', width, height)
else:
    _, frame = camera.read()
    height, width, _ = frame.shape

if SaveVideo:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = round(camera.get(cv2.CAP_PROP_FPS))
    videoWriter = cv2.VideoWriter(
        "/".join(file.split("/")[:-1]) + '/output_{}'.format(file.split("/")[-1]), fourcc, fps, (width, height))

# buffers for demo in batch
buffer_inp = list()
buffer_pre = list()

elapsed = 0
start = timer()
#postprocessed = []
# Loop through frames
n = 0
while camera.isOpened():
    elapsed += 1
    _, frame = camera.read()
    if frame is None:
        print ('\nEnd of Video')
        break
    if FLAGS["skip"] != n :
        n+=1
        continue
    n = 0
    if FLAGS["BK_MOG"] and FLAGS["track"] :
        fgmask = fgbg.apply(frame)
    else :
        fgmask = None
    preprocessed = tfnet.framework.preprocess(frame)
    buffer_inp.append(frame)
    buffer_pre.append(preprocessed)
    # Only process and imshow when queue is full
    feed_dict = {tfnet.inp: buffer_pre}
    net_out = tfnet.sess.run(tfnet.out, feed_dict)
    for img, single_out in zip(buffer_inp, net_out):
        if not FLAGS["track"] :
            postprocessed = tfnet.framework.postprocess(
                single_out, img)
        else :
            postprocessed = tfnet.framework.postprocess(
                single_out, img,frame_id = elapsed,
                csv_file=f,csv=writer,mask = fgmask,
                encoder=encoder,tracker=tracker)
            # print(w.get("radar_info"))
        if SaveVideo:
            videoWriter.write(postprocessed)
        if FLAGS["display"] :
            cv2.imshow('The Postprocessed Video', postprocessed)

        # Clear Buffers
        buffer_inp = list()
        buffer_pre = list()

    if elapsed % 5 == 0:
        sys.stdout.write('\r')
        sys.stdout.write('{0:3.3f} FPS'.format(
            elapsed / (timer() - start)))
        sys.stdout.flush()
    if FLAGS["display"] :
        choice = cv2.waitKey(1)
        if choice == 27:
            break

sys.stdout.write('\n')
if SaveVideo:
    videoWriter.release()
if FLAGS["csv"] :
    f.close()
camera.release()
if FLAGS["display"] :
    cv2.destroyAllWindows()
exit("Bye Bye!")

# FLAGS.demo = "darkflow/sample_vid/multiple.mp4"# video file to use, or if camera just put "camera"
# FLAGS.model = "darkflow/cfg/yolov2-voc-2c.cfg" # tensorflow model
# FLAGS.load = 14300 # tensorflow weights
# # FLAGS.pbLoad = "tiny-yolo-voc-traffic.pb" # tensorflow model
# # FLAGS.metaLoad = "tiny-yolo-voc-traffic.meta" # tensorflow weights
# FLAGS.threshold = 0.4 # threshold of decetion confidance (detection if confidance > threshold )
# FLAGS.gpu = 0.8 #how much of the GPU to use (between 0 and 1) 0 means use cpu
# FLAGS.track = True # wheither to activate tracking or not
# FLAGS.trackObj = ['drone', 'bird'] # the object to be tracked
# #FLAGS.trackObj = ["person"]
# FLAGS.saveVideo = False  #whether to save the video or not
# FLAGS.BK_MOG = False # activate background substraction using cv2 MOG substraction,
#                         #to help in worst case scenarion when YOLO cannor predict(able to detect mouvement, it's not ideal but well)
#                         # helps only when number of detection < 3, as it is still better than no detection.
# FLAGS.tracker = "deep_sort" # wich algorithm to use for tracking deep_sort/sort (NOTE : deep_sort only trained for people detection )
# FLAGS.skip = 0 # how many frames to skipp between each detection to speed up the network
# FLAGS.csv = False #whether to write csv file or not(only when tracking is set to True)
# FLAGS.display = True # display the tracking or not
