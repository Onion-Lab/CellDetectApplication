import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
import os
from cell_processor import *
from collections import Counter

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()
        self.path = ""
        self.x = []
        self.y_volume = []
        self.y_brightness = []
        self.avg_volume = []
        self.avg_brightness = []
        self.draw_flag = False

    def setupUI(self):
        self.setGeometry(600, 200, 1200, 600)
        self.setWindowTitle("Processor")
        self.setWindowIcon(QIcon('icon.png'))

        self.selectFile = QPushButton("Select")
        self.pushButton = QPushButton("Draw")
        self.pushButton2 = QPushButton("Clear")
        self.fileName = QLabel("Not Selected")

        self.selectFile.clicked.connect(self.FileLoad)
        self.pushButton.clicked.connect(self.pushButtonClicked)
        self.pushButton2.clicked.connect(self.clear)

        self.fig, self.ax = plt.subplots(2,1)
        
        self.canvas = FigureCanvas(self.fig)
        
        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.canvas)

        # Right Layout
        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.selectFile)
        rightLayout.addWidget(self.fileName)
        rightLayout.addWidget(self.pushButton)
        rightLayout.addWidget(self.pushButton2)
        rightLayout.addStretch(1)

        layout = QHBoxLayout()
        layout.addLayout(leftLayout)
        layout.addLayout(rightLayout)
        layout.setStretchFactor(leftLayout, 1)
        layout.setStretchFactor(rightLayout, 0)

        self.setLayout(layout)

    def pushButtonClicked(self):
        
        if len(self.x) == 0 :
            QMessageBox.about(self,'ERROR','Not Selected File')
        elif self.draw_flag :
            #self.ax[1].cla()
            #self.ax[0].cla()
            #for i in range(len(self.x)) :
            #    self.ax[0].plot(self.x[i],self.y_volume[i], 'r-',linewidth=0.5)
            #    self.ax[0].set_xlabel("X")
            #    self.ax[0].set_ylabel("volume")

            #for i in range(len(self.x)) :
            #    self.ax[1].plot(self.x[i],self.y_brightness[i], 'b-',linewidth=0.5)
            #    self.ax[1].set_xlabel("X")
            #    self.ax[1].set_ylabel("brightness")

            #self.ax[0].plot([int(i) for i in range(len(self.avg_volume))],self.avg_volume, 'g-',linewidth=3)
            #self.ax[1].plot([int(i) for i in range(len(self.avg_brightness))],self.avg_brightness[i], 'm-',linewidth=3)
            QMessageBox.about(self,'ERROR','Exist Canvas')
        else :
            self.ax[1].cla()
            self.ax[0].cla()
            for i in range(len(self.x)) :
                self.ax[0].plot(self.x[i],self.y_volume[i], 'r-',linewidth=0.5)
                self.ax[0].set_xlabel("X")
                self.ax[0].set_ylabel("volume")

            for i in range(len(self.x)) :
                self.ax[1].plot(self.x[i],self.y_brightness[i], 'b-',linewidth=0.5)
                self.ax[1].set_xlabel("X")
                self.ax[1].set_ylabel("brightness")
            
            self.ax[0].plot([int(i) for i in range(len(self.avg_volume))], self.avg_volume, 'g-',linewidth=3)
            self.ax[1].plot([int(i) for i in range(len(self.avg_brightness))], self.avg_brightness, 'm-',linewidth=3)

        self.draw_flag = True
        self.canvas.draw()
    
    def clear(self):
        if len(self.x) == 0 :
            QMessageBox.about(self,'ERROR','Not Selected File')
        elif self.draw_flag :
            self.ax[1].cla()
            self.ax[0].cla()
            self.draw_flag = False
        else :
            QMessageBox.about(self,'ERROR','Empty Canvas')
        
        self.canvas.draw()

    def nanaverage(self,A,weights,axis):
        return np.nansum(A*weights,axis=axis)/((~np.isnan(A))*weights).sum(axis=axis)

    def FileLoad(self):
        self.x = []
        self.y_volume = []
        self.y_brightness = []
        self.avg_volume = []
        self.avg_brightness = []
        self.draw_flag = False

        fname=QFileDialog.getOpenFileName(self)
        self.path = fname[0]
        self.fileName.setText(self.path.split("/")[-1])

        if ".csv" in self.path.split("/")[-1] :
            result_csv = pd.read_csv(fname[0]).to_dict('list')
        elif ".nd2" in self.path.split("/")[-1] :
            try:
                cellProcessor = CellProcessor()
        
                data = cellProcessor.nd2_to_ndarray(fname[0])
                
                stats = cellProcessor.processing_cell_data(data)
                
                dfStats = cellProcessor.stats_to_dataframe(stats)
                
                if os.path.isfile(fname[0]+'_result.csv'):
                    os.remove(fname[0]+'_result.csv')
                cellProcessor.save_to_csv(dfStats, fname[0]+'_result.csv')
                
            except Exception:
                QMessageBox.about(self,'ERROR','Processing Error')
                return

            result_csv = dfStats.to_dict('list')
        
        try :        
            frame_id = result_csv['frame_id']
            cnt = Counter(frame_id)
            start = 0
            for i in cnt :
                self.x.append(cnt[i])
            for i in range(len(self.x)) :
                self.y_volume.append(np.array([float(i) for i in result_csv['volume'][start:start+self.x[i]]]))
                self.y_brightness.append(np.array([float(i) for i in result_csv['brightness'][start:start+self.x[i]]]))
                start += self.x[i]
                self.x[i] = np.array([int(i) for i in range(self.x[i])])
        except Exception :
            QMessageBox.about(self,'ERROR','Get Volume and Brigtness Error')
            return


        df = pd.DataFrame(self.y_volume)
                
        for i in range(len(df.columns)):
            self.avg_volume.append(self.nanaverage(df[i],1,0))
        
        df = pd.DataFrame(self.y_brightness)
        for i in range(len(df.columns)):
            self.avg_brightness.append(self.nanaverage(df[i],1,0))

        print(pd.DataFrame(self.y_volume), self.avg_volume)
        print(pd.DataFrame(self.y_brightness), self.avg_brightness)

        QMessageBox.about(self,'Success','Success Processing')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()