from __future__ import unicode_literals
import sys, numpy as np
import os
import random
import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets

from numpy import arange, sin, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

progname = os.path.basename(sys.argv[0])
progversion = "0.1"


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='k', edgecolor='c')
        fig.tight_layout(pad=0.01)
        self.axes = fig.add_subplot(111, polar=True)
        self.axes.set_facecolor("k")
        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class MyStaticMplCanvas(MyMplCanvas):
    """Simple canvas with a sine plot."""

    def compute_initial_figure(self):
        self.axes.set_theta_zero_location('N')
        # self.ax.set_theta_direction('clockwise') #If we want it to be clockwise
        self.axes.set_thetamin(-60)
        self.axes.set_thetamax(60)
        self.axes.tick_params(axis="x", colors='white')
        self.axes.tick_params(axis="y", colors='white')
        self.axes.grid(color='w', linestyle='--', linewidth=1, dash_capstyle='butt')
        self.axes.set_rticks([0, 10, 20, 30, 40, 50])
        self.axes.set_xticks(np.linspace(-np.pi / 3, np.pi / 3, 9))
        self.axes.set_title("DEEP SENSE\nRadar View", color='green').set_position([.5, 0.9])


class MyDynamicMplCanvas(MyMplCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, *args, **kwargs):
        MyMplCanvas.__init__(self, *args, **kwargs)
        # timer = QtCore.QTimer(self)
        # timer.timeout.connect(self.update_figure)
        # timer.start()

    def compute_initial_figure(self):
        self.axes.set_xticks([0, 2, 4, 6, 8, 10])
        self.axes.set_yticks([0, 2, 4, 6, 8, 10])
        self.axes.plot(4, 3, color='r', marker='D')

    def update_figure(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        m = random.randint(1, 9)
        l = random.randint(1, 9)
        self.axes.clear()
        self.axes.plot(m, l, color='r', marker='D')
        self.axes.set_xticks([0, 2, 4, 6, 8, 10])
        self.axes.set_yticks([0, 2, 4, 6, 8, 10])
        self.axes.grid(True)
        self.draw()


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

        self.main_widget = QtWidgets.QWidget(self)

        l = QtWidgets.QVBoxLayout(self.main_widget)
        sc = MyStaticMplCanvas(self.main_widget, width=5, height=4, dpi=150)
        self.dc = MyDynamicMplCanvas(self.main_widget, width=5, height=4, dpi=100)
        l.addWidget(sc)
        l.addWidget(self.dc)

        timer = QtCore.QTimer(self)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("All hail matplotlib!", 2000)
        timer.timeout.connect(self.update_figure)
        timer.start(50)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def update_figure(self):
        self.dc.update_figure()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """embedding_in_qt5.py example
Copyright 2005 Florent Rougon, 2006 Darren Dale, 2015 Jens H Nielsen

This program is a simple example of a Qt5 application embedding matplotlib
canvases.

It may be used and modified with no restriction; raw copies as well as
modified versions may be distributed without limitation.

This is modified from the embedding in qt4 example to show the difference
between qt4 and qt5"""
                                )


qApp = QtWidgets.QApplication(sys.argv)

aw = ApplicationWindow()
aw.setWindowTitle("%s" % progname)
aw.show()
sys.exit(qApp.exec_())
#qApp.exec_()