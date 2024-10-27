import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit, QRadioButton, QProgressBar
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Import your custom functions
from visualizer import plot_vectors  # Assuming this function generates the plot
from classification import generate_vectors, train_multiple_classifiers
from utilles import generate_csv_from_txt  # Assuming this function creates CSV from text files


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # File/folder selection section
        self.file_label = QLabel("Select file or folder:")
        self.file_browse_button = QPushButton("Browse Folder")
        self.file_upload_button = QPushButton("Upload File")
        self.file_upload_button.setEnabled(False)
        self.file_text = QTextEdit()
        self.file_text.setReadOnly(True)

        self.layout.addWidget(self.file_label)
        self.layout.addWidget(self.file_browse_button)
        self.layout.addWidget(self.file_upload_button)
        self.layout.addWidget(self.file_text)

        # Radio buttons for upload type
        self.upload_type_group = QHBoxLayout()
        # self.single_file_button = QRadioButton("Single Text File", self)
        # self.single_file_button.setChecked(True)
        # self.folder_button = QRadioButton("Folder of Folders", self)
        # self.upload_type_group.addWidget(self.single_file_button)
        # self.upload_type_group.addWidget(self.folder_button)
        self.layout.addLayout(self.upload_type_group)

        # Process button
        self.process_button = QPushButton("Process")
        self.layout.addWidget(self.process_button)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.hide()  # Hide progress bar initially
        self.layout.addWidget(self.progress_bar)

        # Plot area
        self.plot_label = QLabel("Plot:")
        self.plot_canvas = FigureCanvas(plt.figure())
        self.layout.addWidget(self.plot_label)
        self.layout.addWidget(self.plot_canvas)

        # Connect button clicks to functions
        self.file_browse_button.clicked.connect(self.on_folder_upload_clicked)
        self.file_upload_button.clicked.connect(self.on_file_upload_clicked)
        self.process_button.clicked.connect(self.on_process_clicked)

        # Initialize variables
        self.file_path = None
        self.vectors = None

    def on_folder_upload_clicked(self):
        try:
            self.file_path = QFileDialog.getExistingDirectory(self, "Select folder")
            if self.file_path:
                self.file_text.setText(self.file_path)
        except Exception as e:
            print(f"An error occurred while selecting file: {e}")

    def on_file_upload_clicked(self):
        try:
            self.file_path, _ = QFileDialog.getOpenFileName(self, "Select file", "", "Text Files (*.txt)")
            if self.file_path:
                self.file_text.setText(self.file_path)
        except Exception as e:
            print(f"An error occurred while selecting file: {e}")


    def on_process_clicked(self):
        try:
            if not self.file_path:
                return

            self.plot_canvas.figure.clear()
            self.process_button.hide()  # Hide process button
            self.progress_bar.show()  # Show progress bar

            generate_csv_from_txt("sfarim.csv", folder_path=self.file_path)
            df = pd.read_csv("sfarim.csv")

            self.progress_bar.setMaximum(len(df))

            self.worker = Worker(df)
            self.worker.progress_signal.connect(self.progress_bar.setValue)
            self.worker.finished.connect(self.on_processing_finished)
            self.worker.start()
        except Exception as e:
            print(f"An error occurred while processing: {e}")

    def on_processing_finished(self):
        figure = plot_vectors("vectors.pkl", show_plot= False)  # Call your plotting function
        self.plot_canvas.draw()
        self.progress_bar.hide()  
        self.process_button.show()
        self.file_upload_button.setEnabled(True)

class Worker(QThread):
    progress_signal = pyqtSignal(int)

    def __init__(self, df):
        super().__init__()
        self.df = df

    def run(self):
        generate_vectors(self.df, "name", "content", "vectors.pkl", progress_signal=self.progress_signal)


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"An error occurred while running the application: {e}")
