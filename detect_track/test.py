from PyQt5 import QtCore, QtWidgets, QtMultimedia, QtGui
import sys

app = QtGui.QGuiApplication(sys.argv)
player = QtMultimedia.QMediaPlayer()
sound = QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile("sounds/mmn.mp3"))
player.setMedia(sound)
player.setVolume(50)
player.play()
sys.exit(app.exec_())


# """
# Demo of a line plot on a polar axis.
# """
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# theta = np.pi/4
# r = 30
#
# ax = plt.subplot(111, polar = True)
# ax.set_theta_zero_location('N')
# ax.set_theta_direction('clockwise')
# ax.set_thetamin(-60)
# ax.set_thetamax(60)
# ax.plot(1, 25, color="salmon", label="intruder", marker="D")
# ax.set_rticks([10, 20, 30, 40, 50])
# ax.grid(True)
# ax.set_xticks(np.linspace(-np.pi/3, np.pi/3, 9))
#
# ax.set_title("A line plot on a polar axis")
# plt.show()

# import numpy as np
# import cv2
# import time
# cap = cv2.VideoCapture(0)
# elapsed = 0
# pt1 = (200, 200)
# pt2 = (400, 200)
# stime = time.time()
# K = np.array([[840.47567829, 0., 324.39250648],
#                             [0., 838.77394431, 268.87945384],
#                             [0., 0., 1.]])
# D = np.array([0., 0., 0., 0.])
# Knew = K.copy()
# Knew[(0, 1), (0, 1)] = 1 * Knew[(0, 1), (0, 1)]
# while(cap.isOpened()):
#     elapsed += 1
#     _, frame = cap.read()
#     # frame = cv2.fisheye.undistortImage(frame, K, D, Knew=Knew)
#     # frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
#
#     fps = elapsed / (time.time()-stime)
#     cv2.putText(frame, "FPS: {:.1f}".format(fps), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
#     # if(elapsed % 14 > 7):
#     #     frame = cv2.arrowedLine(frame, pt1, pt2, (0, 0, 254), 4, 4, tipLength=0.5)
#     cv2.imshow("test12", frame)
#     if (cv2.waitKey(1) & 0xFF == ord('q')):
#         break
# cap.release()
#

# import cv2
# cap = cv2.VideoCapture(0)
# while(cap.isOpened()):
#     _, frame = cap.read()
#     frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
#     cv2.imshow("test12", frame)
#     if (cv2.waitKey(1) & 0xFF == ord('q')):
#         break
# cap.release()

# def cam_to_world_x(y):
#     return (5.73 * y + 320)
#
# print(cam_to_world_x(0.8716))

# from matplotlib.figure import Figure
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# import matplotlib.gridspec as gridspec
#
# fig = Figure(figsize=(3.5, 3.5), frameon=False)
# canvas = FigureCanvas(fig)
# gs = gridspec.GridSpec(1, 1,
#                        left=0.15, right=0.95, bottom=0.15, top=0.95,
#                        wspace=None, hspace=None,
#                        width_ratios=None, height_ratios=None)
# ax = fig.add_subplot(gs[0])
# canvas.print_figure("abc" + ".png", format="png")




# import numpy as np
# import cv2
#
# img = cv2.imread("images/watermark.png", cv2.IMREAD_UNCHANGED)
# img = cv2.resize(img, (80, 80))
# img = cv2.flip(img, 1)
# (B, G, R, A) = cv2.split(img)
#
# B = cv2.bitwise_and(B, B, mask=A)
# G = cv2.bitwise_and(G, G, mask=A)
# R = cv2.bitwise_and(R, R, mask=A)
# watermark = cv2.merge([B, G, R, A])
#
# cv2.imwrite("images/watermarkW80x80.png", img)



# import numpy as np
# import cv2 as cv
# def alertSystem(isalert=1, transparency=0, lowLoop=False, highLoop=False):
#     if (isalert):
#         if round((transparency-1), 3) == 0:
#             lowLoop = True
#             highLoop = False
#             transparency -= 0.1
#         if round(transparency, 3) == 0:
#             highLoop = True
#             lowLoop = False
#             transparency += 0.1
#         elif (lowLoop):
#             transparency -= 0.1
#         elif (highLoop):
#             transparency += 0.1
#         # print(transparency)
#         return (transparency, lowLoop, highLoop)
#
# if __name__ == "__main__":
#     cy = 50
#     cx = 50
#     trans, ll, hl = 0, False, False
#     watermark = cv.imread("images/watermark.png", cv.IMREAD_UNCHANGED)
#     (wH, wW) = watermark.shape[:2]
#     # (B, G, R, A) = cv.split(watermark)
#     # B = cv.bitwise_and(B, B, mask=A)
#     # G = cv.bitwise_and(G, G, mask=A)
#     # R = cv.bitwise_and(R, R, mask=A)
#     # watermark = cv.merge([B, G, R, A])
#     # Creating the image's overlay with the watermark
#     overlay = np.zeros((480, 640, 4), dtype="uint8")
#     overlay[cy:wH + cy, cx:wW + cx] = watermark
#     # print(overlay)
#     # cv.imshow("test", overlay)
#     # Applying the overlay
#     cap = cv.VideoCapture(0)
#     elapsed = 0
#     while(cap.isOpened()):
#         _, frame = cap.read()
#         (trans, ll, hl) = alertSystem(isalert=1, transparency = trans, lowLoop=ll, highLoop=hl)
#         frame = np.dstack([frame, np.ones((480, 640), dtype="uint8") * 255])
#         # print(overlay.shape, output.shape)
#         cv.addWeighted(overlay, trans, frame, 1.0, 0, frame)
#         print(frame.shape)
#         cv.imshow("out", frame)
#         if (cv.waitKey(1) & 0xFF == ord('q')):
#             break
#     cap.release()



# from __future__ import division
# import numpy as np
# from numpy import pi
# import matplotlib.pyplot as pp
#
# # normalize and convert to dB
# dbnorm = lambda x: 20*np.log10(np.abs(x)/np.max(x));
#
# # generate example data
# # some angles
# alpha = np.arange(-90, 90, 0.01);
# x = np.deg2rad(alpha)
# dir_function = dbnorm(np.sinc(x))
#
# # plot
# ax = pp.subplot(111, polar=True)
# # set zero north
# ax.set_theta_zero_location('N')
# ax.set_theta_direction('clockwise')
# pp.plot(np.deg2rad(alpha), dir_function)
# ax.set_ylim(-20,0)
# ax.set_yticks(np.array([-20, -12, -6, 0]))
# ax.set_xticks(np.array([0, -45, -90, np.nan, np.nan, np.nan, 90, 45])/180*pi)
# pp.show()