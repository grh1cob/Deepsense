import sys
import os

from PyQt5.QtCore import QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import cv2
from time import time as timer
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi

class GUIDEEP(QDialog):
    def __init__(self, options):
        super(GUIDEEP, self).__init__()
        loadUi('layout.ui', self)
        self.options = options
        self.image = None
        self.postImage = None
        self.filename = options["demo"]
        self.SaveVideo = options["saveVideo"]
        self.elapsed = 0
        self.skipnum = 0
        self.track = options["track"]
        # if options["track"]:
        #     if options["tracker"] == "deep_sort":
        #         from deep_sort import generate_detections
        #         from deep_sort.deep_sort import nn_matching
        #         from deep_sort.deep_sort.tracker import Tracker
        #         metric = nn_matching.NearestNeighborDistanceMetric(
        #             "cosine", 0.2, 100)
        #         self.tracker = Tracker(metric)
        #         self.encoder = generate_detections.create_box_encoder(
        #             os.path.abspath("deep_sort/resources/networks/mars-small128.ckpt-68577"))
        #     elif options["tracker"] == "sort":
        #         from sort.sort import Sort
        #         self.encoder = None
        #         self.tracker = Sort()
        # if options["BK_MOG"] and options["track"]:
        #     fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

        # assert os.path.isfile(self.filename), \
        # 'file {} does not exist'.format(self.filename)

        self.camera = cv2.VideoCapture(self.filename)

        assert self.camera.isOpened(), \
            'Cannot capture source'

        if self.SaveVideo:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.fps = round(self.camera.get(cv2.CAP_PROP_FPS))
            self.videoWriter = cv2.VideoWriter(
                "/".join(self.filename.split("/")[:-1]) + '/output_{}'.format(self.filename.split("/")[-1]), fourcc, fps, (1280, 720))

        self.initNetButton.clicked.connect(self.initTFNet)
        self.startButton.clicked.connect(self.start_cam)
        self.stopButton.clicked.connect(self.stop_cam)


    def initTFNet(self):
        pass

    @pyqtSlot()
    def start_cam(self):
        self.timer = QTimer(self)
        print("hi1")
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        ret, self.postImage = self.camera.read()
        self.displayImage()

    def stop_cam(self):
        self.timer.stop()
        # self.camera.release()

    def displayImage(self):
        qformat = QImage.Format_Indexed8
        qformat = QImage.Format_RGB888
        # if len(self.postImage.shape) == 3:
        #     if(self.postImage.shape[2]) == 4:
        #         qformat = QImage.Format_RGBA8888
        #     else:
        #         qformat = QImage.Format_RGB888
        self.outImage = QImage(self.postImage, self.postImage.shape[1], self.postImage.shape[0], self.postImage.strides[0], qformat).rgbSwapped()
        self.vidLabel.setPixmap(QPixmap.fromImage(self.outImage))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    FLAGS = {
        'demo': 0,
        'model': 'darkflow/cfg/yolov2-voc-2c.cfg',
        'load': 14300,
        'threshold': 0.3,
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
    window = GUIDEEP(FLAGS)
    window.setWindowTitle("Deep Sense")
    window.show()
    sys.exit(app.exec_())