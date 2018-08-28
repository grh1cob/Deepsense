import random
import sys
import os
import socket
import numpy as np
from PyQt5 import QtGui, QtMultimedia
from PyQt5.QtCore import QTimer, pyqtSlot, QSize, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QImage, QPixmap, QLinearGradient, QPalette, QColor, QBrush
import cv2
from piyush import Connect
from time import time as timer
from PyQt5.QtWidgets import QDialog, QApplication, QSizePolicy, QLabel
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import csv

class wifi_con(object):
    def __init__(self, hostname = '127.0.0.1', port = 6785):
        self.host=hostname
        self.port=port
        try:
            self.socketcreated = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socketcreated.settimeout(1)
        except socket.error:
            print("unable to wiFi connect")
            self.socketcreated.close()
        try:
            self.socketcreated.bind((self.host, self.port))
            print("Connection Successfull")
        except socket.error:
            print("unable to open the WiFi port")
            self.socketcreated.close()
    def wifi_read(self, size=24):
        self.rxdata = self.socketcreated.recv(size)
        return self.rxdata
    def wifi_write(self, txdata):
        self.socketcreated.send(txdata)
    def wifi_close(self):
        self.socketcreated.close()
    def __del__(self):
        self.socketcreated.close()


class DEEPGUI(QDialog):
    def __init__(self, options):
        super(DEEPGUI, self).__init__()
        # Load UI from .ui file to self. (Everything to be referenced from now in the sense of UI will be assigned to self only intentionally)
        # (Loading to any other variable is also possible)
        loadUi('layout.ui', self)
        self.initPalette()
        self.initButtons()
        self.initSounds()

        self.plotskip = 2
        file = open("radar_data.csv", "r")
        self.reader = list(csv.reader(file))
        print(self.reader)
        # TFNet variables to be used
        self.options = options
        self.postImage = None
        self.filename = options["demo"]
        self.SaveVideo = options["saveVideo"]
        self.elapsed = 0
        self.skipnum = options["skip"]
        self.track = options["track"]
        (self.vidframewidth, self.vidframeheight) = (640, 480)

        self.plotskip = 1

        # Initialize Capture Components
        self.initCapture()

        # Initialize the overlay Image
        self.overlayE = None
        self.overlayW = None
        self.overlayN = None
        self.overlayS = None
        self.initOverlay()

        # Alerting Parameters
        self.tp, self.ll, self.hl, self.overl = 0, False, False, None

        # ProgressBar Initial Value
        self.progressBar.setValue(0)

        # Initialization Of Canvas Plot
        self.dc = MyMplCanvas(self, width=5, height=4, dpi=120)
        self.dc.move(660, 70)

    def initPalette(self):
        oImage = QImage("images/background.jpg")
        sImage = oImage.scaled(self.frameSize())  # resize Image to QDialog's size

        # Initialize the QPalette for beautifying
        palette = QPalette()
        # palette.setColorGroup(0, QColor('white'), QColor('cyan'), QColor('white'), QColor('cyan'),
        #                       QColor('cyan'), QColor('black'), QColor('red'), QColor('cyan'), QBrush(sImage))

        palette.setBrush(QPalette.Background, QBrush(sImage))
        palette.setColor(QPalette.Foreground, QColor("white"))
        self.setPalette(palette)

    # Initializing the Vision Sensor Capture
    def initCapture(self):
        # Initialize with the camera or the Video File
        if self.filename == 'cam':
            self.filename = 0
        else:
            assert os.path.isfile(self.filename), \
                'file {} does not exist'.format(self.filename)
        self.camera = cv2.VideoCapture(self.filename)

        assert self.camera.isOpened(), \
            'Cannot capture source'

        # Initialize VideoWriter for saving video
        if self.SaveVideo:
            if not self.filename == "cam":
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.fps = round(self.camera.get(cv2.CAP_PROP_FPS))
                self.videoWriter = cv2.VideoWriter(
                    "/".join(self.filename.split("/")[:-1]) + '/output_{}'.format(self.filename.split("/")[-1]), fourcc,
                    self.fps, (self.vidframewidth, self.vidframeheight))
            else:
                self.fps =  None

    # Button Custoizations and Initializations
    def initButtons(self):
        self.initNetButton.setStyleSheet(
            'background-color: #e6dc20; border-style: outset;border-width: 2px;border-radius: 10px;border-color: beige;font: bold 14px;min-width: 10em;padding: 6px;')
        self.startButton.setStyleSheet(
            'background-color: #45c71a; border-style: outset;border-width: 2px;border-radius: 10px;border-color: beige;font: bold 14px;min-width: 10em;padding: 6px;')
        self.stopButton.setStyleSheet(
            'background-color: #d15843; border-style: outset;border-width: 2px;border-radius: 10px;border-color: beige;font: bold 14px;min-width: 10em;padding: 6px;')
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.initNetButton.clicked.connect(self.initTrack)
        self.startButton.clicked.connect(self.start_cam)
        self.stopButton.clicked.connect(self.stop_cam)
        self.exitButton.clicked.connect(self.closeUI)

    # For initializing the sounds of Traffic Advisory and load them into player accordingly.
    def initSounds(self):
        self.player = QtMultimedia.QMediaPlayer()
        self.alertSoundA = QtMultimedia.QMediaContent(QUrl.fromLocalFile("sounds/cr.mp3"))
        self.alertSoundB = QtMultimedia.QMediaContent(QUrl.fromLocalFile("sounds/mmn.mp3"))
        self.alertSoundC = QtMultimedia.QMediaContent(QUrl.fromLocalFile("sounds/ta.mp3"))
        self.alertSoundD = QtMultimedia.QMediaContent(QUrl.fromLocalFile("sounds/tt.mp3"))

    def playSound(self, name):
        self.player.stop()
        self.player.setMedia(name)
        self.player.setVolume(80)
        self.player.play()

    def initTrack(self):
        if (self.track):
            self.progressBar.setValue(20)
            if self.options["track"]:
                if self.options["tracker"] == "deep_sort":
                    from deep_sort import generate_detections
                    from deep_sort.deep_sort import nn_matching
                    from deep_sort.deep_sort.tracker import Tracker
                    self.progressBar.setValue(50)
                    metric = nn_matching.NearestNeighborDistanceMetric(
                        "cosine", 0.2, 100)
                    self.tracker = Tracker(metric)
                    self.encoder = generate_detections.create_box_encoder(
                        os.path.abspath("deep_sort/resources/networks/mars-small128.ckpt-68577"))
                elif self.options["tracker"] == "sort":
                    from sort.sort import Sort
                    self.encoder = None
                    self.tracker = Sort()
                    self.progressBar.setValue(50)
            if self.options["BK_MOG"] and self.options["track"]:
                fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
            self.progressBar.setValue(60)
            self.initTFNet()
        else:
            self.initTFNet()

    def initTFNet(self):
        from darkflow.darkflow.net.build import TFNet
        self.tfnet = TFNet(self.options)
        self.progressBar.setValue(100)
        print("done")
        self.startButton.setEnabled(True)

    def initOverlay(self):
        watermarkE = cv2.imread("images/watermarkE80x80.png", cv2.IMREAD_UNCHANGED)
        (B, G, R, A) = cv2.split(watermarkE)
        B = cv2.bitwise_and(B, B, mask=A)
        G = cv2.bitwise_and(G, G, mask=A)
        R = cv2.bitwise_and(R, R, mask=A)
        watermarkE = cv2.merge([B, G, R, A])
        watermarkW = cv2.imread("images/watermarkW80x80.png", cv2.IMREAD_UNCHANGED)
        (B, G, R, A) = cv2.split(watermarkW)
        B = cv2.bitwise_and(B, B, mask=A)
        G = cv2.bitwise_and(G, G, mask=A)
        R = cv2.bitwise_and(R, R, mask=A)
        watermarkW = cv2.merge([B, G, R, A])
        watermarkN = cv2.imread("images/watermarkN80x80.png", cv2.IMREAD_UNCHANGED)
        (B, G, R, A) = cv2.split(watermarkN)
        B = cv2.bitwise_and(B, B, mask=A)
        G = cv2.bitwise_and(G, G, mask=A)
        R = cv2.bitwise_and(R, R, mask=A)
        watermarkN = cv2.merge([B, G, R, A])
        watermarkS = cv2.imread("images/watermarkS80x80.png", cv2.IMREAD_UNCHANGED)
        (B, G, R, A) = cv2.split(watermarkS)
        B = cv2.bitwise_and(B, B, mask=A)
        G = cv2.bitwise_and(G, G, mask=A)
        R = cv2.bitwise_and(R, R, mask=A)
        watermarkS = cv2.merge([B, G, R, A])
        (wH, wW) = watermarkE.shape[:2] # Assuming all have same dimension
        (h, w) = (self.vidframeheight, self.vidframewidth)

        # Creating the image's overlays with the watermark
        overlayE = np.zeros((h, w, 4), dtype="uint8")
        overlayE[int((h - wH) / 2):int((h + wH) / 2), w - wW - 5:w - 5] = watermarkE

        overlayW = np.zeros((h, w, 4), dtype="uint8")
        overlayW[int((h - wH) / 2):int((h + wH) / 2), 5:wW + 5] = watermarkW

        overlayN = np.zeros((h, w, 4), dtype="uint8")
        overlayN[5:wH + 5, int((w - wW) / 2):int((w + wW) / 2)] = watermarkN

        overlayS = np.zeros((h, w, 4), dtype="uint8")
        overlayS[h - wH - 5:h - 5, int((w - wW) / 2):int((w + wW) / 2)] = watermarkS

        (self.overlayE, self.overlayW, self.overlayN, self.overlayS) = (overlayE, overlayW, overlayN, overlayS)

    def alertSystem(self, isalert=1, transparency=0, lowLoop=False, highLoop=False, dir="E"):
        if(dir == "E"):
            overlay = self.overlayE
        elif(dir == "W"):
            overlay = self.overlayW
        elif (dir == "N"):
            overlay = self.overlayN
        elif (dir == "S"):
            overlay = self.overlayS
        if (isalert):
            if round((transparency - 1), 3) == 0:
                lowLoop = True
                highLoop = False
                transparency -= 0.1
            if round(transparency, 3) == 0:
                highLoop = True
                lowLoop = False
                transparency += 0.1
            elif (lowLoop):
                transparency -= 0.1
            elif (highLoop):
                transparency += 0.1
            return(transparency, lowLoop, highLoop, overlay)
        else:
            self.tp, self.ll, self.hl, self.overl = 0, False, False, None

    def cam_to_world_x(self, x, y):
        return (-324.3925*x + 840.4756*y)

    @pyqtSlot()
    def start_cam(self):
        self.stopButton.setEnabled(True)
        self.startButton.setEnabled(False)

        self.conn = wifi_con()
        self.start_plot()
        self.isalert = 0
        self.isSound = 1

        self.timer = QTimer(self)
        # # For undistorting camera image
        # self.K640_480 = np.array([[840.47567829, 0., 324.39250648],
        #                     [  0., 838.77394431, 268.87945384],
        #                     [  0., 0., 1.]])
        # self.D640_480 = np.array([0., 0., 0., 0.])
        # self.Knew = self.K640_480.copy()
        # self.Knew[(0, 1), (0, 1)] = 1 * self.Knew[(0, 1), (0, 1)]
        self.timer.timeout.connect(self.update_frame)
        self.stime = timer()
        self.timer.start(5)

    def start_plot(self):
        self.timer1 = QTimer(self)
        self.el = 0
        self.mmn = 1
        self.label_3.setStyleSheet("QLabel { color: 'red'; font: bold 28px;}")
        self.label_4.setStyleSheet("QLabel { color: 'red'; font: bold 28px;}")
        self.skipplot = 0
        self.stime1 = timer()
        self.timer1.timeout.connect(self.update_plot)
        self.timer1.start(30)

    def decode_bytes(self, bytearray):
        return(int.from_bytes(bytearray, byteorder='big'))
    def update_plot(self):
        if self.skipplot != self.plotskip:
            self.skipplot += 1
            return
        self.skipplot = 0
        self.el += 1
        range, angle, head, ele = float(self.reader[self.el][0]), float(self.reader[self.el][1]), float(self.reader[self.el][2]), float(self.reader[self.el][3])
        self.label_3.setText("{:.1f}°".format(ele))
        self.label_4.setText("{:.1f}°".format(head))
        if head != 0 or ele != 0:
            self.isalert = 1
            if(head > 0): self.directionEGO = "W"
            elif(head < 0): self.directionEGO = "E"
            elif(ele > 0): self.directionEGO = "N"
            else: self.directionEGO = "S"
        self.dc.update_figure(angle, range)
        # try:
        #     data0 = self.conn.wifi_read(20)
        #     data1 = self.conn.wifi_read(20)
        #     data2 = self.conn.wifi_read(20)
        #     i = self.decode_bytes(data2[16:20])
        #     head, ele = self.decode_bytes(data2[8:12]), self.decode_bytes(data2[12:16])
        #     if (head != 0 or ele != 0):
        #         range, angle = self.decode_bytes(data2[:4]), self.decode_bytes(data2[4:8])
        #         if (angle > 2147483648): angle -= 4294967296
        #         if (head > 2147483648): head -= 4294967296
        #         if (ele > 2147483648): ele -= 4294967296
        #         i = self.decode_bytes(data2[16:20])
        #     else:
        #         head1, ele1 = self.decode_bytes(data1[8:12]), self.decode_bytes(data1[12:16])
        #         if (head1 != 0 or ele1 != 0):
        #             range, angle = self.decode_bytes(data1[:4]), self.decode_bytes(data1[4:8])
        #             if (angle > 2147483648): angle -= 4294967296
        #             if (head1 > 2147483648): head = head1 - 4294967296
        #             if (ele1 > 2147483648): ele = ele1 - 4294967296
        #             i = self.decode_bytes(data1[16:20])
        #         else:
        #             head0, ele0 = self.decode_bytes(data0[8:12]), self.decode_bytes(data0[12:16])
        #             if (head0 != 0 or ele0 != 0):
        #                 range, angle = self.decode_bytes(data0[:4]), self.decode_bytes(data0[4:8])
        #                 if (angle > 2147483648): angle -= 4294967296
        #                 if (head0 > 2147483648): head = head0 - 4294967296
        #                 if (ele0 > 2147483648): ele = ele0 - 4294967296
        #                 i = self.decode_bytes(data0[16:20])
        #             else:
        #                 range, angle = self.decode_bytes(data2[:4]), self.decode_bytes(data2[4:8])
        #                 if (angle > 2147483648): angle -= 4294967296
        #                 if (head > 2147483648): head -= 4294967296
        #                 if (ele > 2147483648): ele -= 4294967296
        #                 i = self.decode_bytes(data2[16:20])
        #
        #     # if(range > 15):
        #     #     self.mmn = 1
        #     #     self.isalert = 0
        #     # print(range, angle, head, ele, i)
        #     range, angle, head, ele = self.decode_bytes(data0[:4]), self.decode_bytes(data0[4:8]), self.decode_bytes(data0[8:12]), self.decode_bytes(data0[12:16])
        #     if(self.el%50 == 0):
        #         self.conn.wifi_close()
        #         self.conn = wifi_con()
        #     # Show heading and elevation change to GUI Labels
        #     self.label_3.setText("{:.1f}°".format(ele/1000))
        #     self.label_4.setText("{:.1f}°".format(head/1000))
        #     if head != 0 or ele != 0:
        #         self.isalert = 1
        #         if(head > 0): self.directionEGO = "W"
        #         elif(head < 0): self.directionEGO = "E"
        #         elif(ele > 0): self.directionEGO = "N"
        #         else: self.directionEGO = "S"
        #     self.dc.update_figure(angle/1000, range/1000)
        #     print(self.el/(timer()-self.stime1))
        # except:
        #     print("except")
        #     pass

    def update_frame(self):

        self.elapsed += 1
        buffer_inp, buffer_pre = list(), list()
        ret, frame = self.camera.read()
        if frame is None:
            print('\nEnd of Video')
            return
        if self.options["skip"] != self.skipnum:
            self.skipnum += 1
            return
        self.skipnum = 0
        # frame = cv2.fisheye.undistortImage(frame, self.K640_480, self.D640_480, Knew=self.Knew) #If you want to apply Undistortion to Images
        preprocessed = self.tfnet.framework.preprocess(frame)
        buffer_inp.append(frame)
        buffer_pre.append(preprocessed)

        # Only process and imshow when queue is full
        feed_dict = {self.tfnet.inp: buffer_pre}
        net_out = self.tfnet.sess.run(self.tfnet.out, feed_dict)
        for img, single_out in zip(buffer_inp, net_out):
            if not self.track:
                postImage = self.tfnet.framework.postprocess(
                    single_out, img)
            else:
                postImage, bbox = self.tfnet.framework.postprocess(
                    single_out, img, frame_id=self.elapsed, mask=None,
                    encoder=self.encoder, tracker=self.tracker, ranger=2)
                self.fps = self.elapsed/(timer()-self.stime)
                postImage = cv2.putText(postImage, "FPS: {:.1f}".format(self.fps), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                if self.isalert:
                    if self.isSound:
                        self.playSound(self.alertSoundB)
                        self.isSound = 0
                        self.mmn = 0
                    postImage = np.dstack([postImage, np.ones((480, 640), dtype="uint8") * 255])
                    self.tp, self.ll, self.hl, self.overl = self.alertSystem(transparency=self.tp, lowLoop=self.ll, highLoop=self.hl, dir=self.directionEGO)
                    self.postImage = cv2.addWeighted(self.overl, self.tp, postImage, 1.0, 0)
                else:
                    self.postImage = postImage
                # print(bbox)
                self.displayImage()

            if self.SaveVideo:
                self.videoWriter.write(self.postImage)

    def stop_cam(self):
        self.conn.wifi_close()
        self.timer.stop()
        self.timer1.stop()
        self.mmn = 1
        self.elapsed = 0
        self.startButton.setEnabled(True)


    def closeUI(self):
        self.camera.release()
        self.close()

    def displayImage(self):
        qformat = QImage.Format_Indexed8
        if len(self.postImage.shape) == 3:
            if (self.postImage.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        self.outImage = QImage(self.postImage, self.postImage.shape[1], self.postImage.shape[0],
                               self.postImage.strides[0], qformat).rgbSwapped()
        self.vidLabel.setPixmap(QPixmap.fromImage(self.outImage))


class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='k', edgecolor='c')
        fig.tight_layout(pad=0.01)
        self.axes = fig.add_subplot(111, polar=True)
        self.axes.set_facecolor("k")
        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,QSizePolicy.Expanding,QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        self.axes.set_theta_zero_location('N')
        # self.ax.set_theta_direction('clockwise') #If we want it to be clockwise
        self.axes.set_thetamin(-60)
        self.axes.set_thetamax(60)
        self.axes.tick_params(axis="x", colors='white')
        self.axes.tick_params(axis="y", colors='white')
        self.axes.grid(color='w', linestyle='--', linewidth=1)
        self.axes.set_rticks([0, 10, 20, 30, 40, 50])
        self.axes.set_xticks(np.linspace(-np.pi / 3, np.pi / 3, 9))
        self.axes.set_title("DEEP SENSE\nRadar View", color='green').set_position([.5, 0.9])

    def update_figure(self, phi=0, rad=0):
        self.axes.clear()
        self.axes.set_theta_zero_location('N')
        # self.ax.set_theta_direction('clockwise') #If we want it to be clockwise
        self.axes.set_thetamin(-60)
        self.axes.set_thetamax(60)
        self.axes.plot(phi, rad, color='r', marker=6, label='Intruder Position', markersize=8)
        self.axes.set_rticks([0, 10, 20, 30, 40, 50])
        self.axes.set_xticks(np.linspace(-np.pi / 3, np.pi / 3, 9))
        self.axes.set_title("DEEP SENSE\nRadar View", color='green').set_position([.5, 0.9])
        self.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    FLAGS = {
        'demo': "test_case_1.mp4", # 'cam' for capture web camera
        'pbLoad': 'darkflow/built_graph/yolov2-voc-2c.pb',
        'metaLoad':'darkflow/built_graph/yolov2-voc-2c.meta',
        'skip': 1,
        'track': True,
        'trackObj': ['drone', 'bird'],
        'tracker': 'deep_sort',
        'BK_MOG': False,
        'saveVideo': False,
        'threshold': 0.3,
        'gpu': 0.7,
        'csv': False
    }
    window = DEEPGUI(FLAGS)
    window.setWindowTitle("Deep Sense")
    app.setWindowIcon(QtGui.QIcon('images/logo.png'))
    window.show()
    sys.exit(app.exec_())