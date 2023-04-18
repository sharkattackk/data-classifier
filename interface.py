import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
import pandas as pd
import numpy as np
import csv
from joblib import load 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
import cleanup
#Ui_MainWindow, QMainWindow = loadUiType("mainwindow.ui")

def extract_features(window):
    features = []
    for col in window.columns:
        features.append(window[col].mean())
        features.append(window[col].median())
        features.append(window[col].min())
        features.append(window[col].max())
        features.append(window[col].var())
        features.append(window[col].max() - window[col].min())
        features.append(skew(window[col]))
        features.append(window[col].std())
        features.append(np.sqrt(np.mean(window[col]**2)))
        features.append(window[col].kurtosis())
    return features

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set window size and title
        self.setWindowTitle("Jump and Walk Classifier")
        self.setGeometry(700, 700, 600, 400)
        # Create labels and buttons
        self.label = QLabel(self)
        self.label.setGeometry(110, 50, 500, 50)
        self.chosen_file = None
        self.prediction = []
        self.label.setText("Choose a CSV file to classify")
        self.btn_browse = QPushButton("Browse", self)
        self.btn_browse.setGeometry(110, 130, 100, 50)
        self.btn_browse.clicked.connect(self.browse_file)
        self.btn_classify = QPushButton("Run", self)
        self.btn_classify.setGeometry(250, 130, 100, 50)
        self.btn_classify.clicked.connect(self.getPred)
        self.btn_classify.setEnabled(False)
        self.btn_csv = QPushButton("Get CSV", self)
        self.btn_csv.setGeometry(390, 130, 100, 50)
        self.btn_csv.clicked.connect(self.getCSV)
        self.btn_csv.setEnabled(False)

        # Create pixmap for displaying plot
        self.pixmap = QPixmap()
        self.plot_label = QLabel(self)
        self.plot_label.setGeometry(50, 200, 500, 150)

    def browse_file(self):
        # Open file dialog to choose a CSV filee
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Choose CSV file", "", "CSV Files (*.csv)", options=options)
        if fileName:
            self.csv_file = fileName
            self.label.setText("Selected file: {}".format(fileName))
        self.chosen_file = fileName
        self.btn_classify.setEnabled(True)
    

    def getPred(self):
            file = pd.read_csv(self.chosen_file)
            file = cleanup.clean(file)
            window_size = 5*100 #100 HZ sample rate
            windows=[] #empty array for storing the Jumping windows
            for i in range(0, len(file)-window_size, window_size):
                window = file[i:i+window_size]
                windows.append(window) #store current window in array
            np.random.shuffle(windows) #shuffle the window


            file_features = []
            for window in windows:
                window_features = extract_features(window)
                file_features.append(window_features)

            clf = load('./models/walk-jump-model.joblib')
            scaler = clf.named_steps['standardscaler']
            file_features_scaled = scaler.transform(file_features)
            prediction = clf.predict(file_features_scaled)
            self.prediction = prediction
            jump_count = 0
            walk_count = 0
            for val in prediction:
                if(val == 1):
                    jump_count += 1
                else:
                    walk_count +=1

            if(jump_count > walk_count):
                output = "Jumping"
            else:
                output = "Walking"

            self.label.setText("This file was recorded while person was {}".format(output))
            self.btn_csv.setEnabled(True)

    def getCSV(self):
        with open('./output/output.csv', mode='w', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(['Window', 'Action'])
            for i in range(len(self.prediction)):
                if self.prediction[i] == 1:
                    writer.writerow([i, 'jumping'])
                else:
                    writer.writerow([i, 'walking'])
        self.label.setText("CSV file Saved as /output/output.csv")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
    
    