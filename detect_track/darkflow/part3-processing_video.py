import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import os

FILE_OUTPUT = 'output.avi'

if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)

option = {
    'model': 'cfg/yolov2-voc.cfg',
    'load': 7625,
    'threshold': 0.26,
    'gpu': 0.8
}

tfnet = TFNet(option)
filepath = "sample_vid/cut/moviemaker.mp4"
capture = cv2.VideoCapture(filepath)
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter(
    "/".join(filepath.split("/")[:-1]) + '/output_{}'.format(filepath.split("/")[-1]), fourcc, 29, (1280, 720))
print("/".join(filepath.split("/")[:-1]))
#out = cv2.VideoWriter('output.avi', -1, 20.0, (1280,720))
# fourcc = cv2.cv.CV_FOURCC(*'X264')
# out = cv2.VideoWriter('FILE_OUTPUT,fourcc', 20.0, (1280,720))

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            text = '{}: {:.0f}%'.format(label, confidence * 100)
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        videoWriter.write(frame)
        #out.write(frame)
        #print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        videoWriter.release()
        cv2.destroyAllWindows()
        break
