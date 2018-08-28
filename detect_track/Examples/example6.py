from PyQt5 import QtWidgets
from PyQt5.QtCore import *
import sys
import time


class TTT(QThread):
    def __init__(self):
        super(TTT, self).__init__()
        self.quit_flag = False

    def run(self):
        while True:
            if not self.quit_flag:
                self.doSomething()
                time.sleep(1)
            else:
                break

        self.quit()
        self.wait()

    def doSomething(self):
        print('123')


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.btn = QtWidgets.QPushButton('run process')
        self.btn.clicked.connect(self.create_process)
        self.setCentralWidget(self.btn)

    def create_process(self):
        if self.btn.text() == "run process":
            print("Started")
            self.btn.setText("stop process")
            self.t = TTT()
            self.t.start()
        else:
            self.t.quit_flag = True
            print("Stop sent")
            self.t.wait()
            print("Stopped")
            self.btn.setText("run process")


if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())