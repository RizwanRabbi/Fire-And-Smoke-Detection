import sys
import os
import cv2
import glob
from PyQt5.QtCore import QTimer, pyqtSignal, QThread
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QProgressBar, QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, QSpacerItem, QSizePolicy, QFileDialog, QLabel
from PyQt5.QtGui import QPixmap, QImage
from ultralytics import YOLO
class VideoProcessingThread(QThread):
    progress_updated = pyqtSignal(int)
    processing_finished = pyqtSignal(str)

    def __init__(self, video_path, model):
        super().__init__()
        self.video_path = video_path
        self.model = model

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = "output_video.mp4"
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference with YOLO model
            results = self.model.predict(source=frame, save=False, conf=0.25)

            # Draw bounding boxes and labels on the frame
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates
                    conf = box.conf[0]  # Confidence
                    cls = int(box.cls[0])  # Class
                    label = f"{self.model.names[cls]}: {conf:.2f}"

                    # Debug print statements
                    print(f"Box coordinates: {x1, y1, x2, y2}")
                    print(f"Label: {label}")

                    # Draw rectangle and label on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # Ensure the text is drawn within the frame
                    y1_text = max(0, y1 - 10)
                    cv2.putText(frame, label, (x1, y1_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Additional debug info: Show frame with label
                    # cv2.imshow("Frame with Label", frame)
                    # cv2.waitKey(1)

            out.write(frame)
            frame_count += 1
            progress_percentage = int((frame_count / total_frames) * 100)
            self.progress_updated.emit(progress_percentage)

        cap.release()
        out.release()
        self.processing_finished.emit(output_path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the background image to get its size
        pixmap = QPixmap("background.png")
        self.background_width = pixmap.width()
        self.background_height = pixmap.height()

        self.setWindowTitle("PyQt5 Button Example")

        # Set fixed window size based on background image dimensions
        self.setFixedSize(self.background_width, self.background_height)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Set background image
        self.central_widget.setStyleSheet("background-image: url(background.png);")

        main_layout = QVBoxLayout()

        # Add a spacer item at the top to push the buttons to the center vertically
        spacer_top = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        main_layout.addItem(spacer_top)

        button_layout = QVBoxLayout()

        # Create three buttons with smaller width, square shape, rounded corners, and specific colors
        button_style = """
        QPushButton {
            width: 150px; 
            height: 90px; 
            border-radius: 10px; 
            background: #45214A;
            color: white;
            font-size: 14px;
            font-weight: bold;
            border: none;
        }
        QPushButton:hover {
            background: #323050;
        }
        """

        self.button1 = QPushButton("Upload Image", self)
        self.button1.setStyleSheet(button_style)
        self.button1.clicked.connect(self.on_button1_click)
        button_layout.addWidget(self.button1)

        self.button2 = QPushButton("Upload Video", self)
        self.button2.setStyleSheet(button_style)
        self.button2.clicked.connect(self.on_button2_click)
        button_layout.addWidget(self.button2)

        self.button3 = QPushButton("Open Webcam", self)
        self.button3.setStyleSheet(button_style)
        self.button3.clicked.connect(self.start_webcam)
        button_layout.addWidget(self.button3)
        self.capture = None
        self.timer = QTimer()

        # Add horizontal spacers to center the buttons horizontally
        hbox = QHBoxLayout()
        hbox.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Minimum))
        hbox.addLayout(button_layout)
        hbox.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Expanding, QSizePolicy.Minimum))

        main_layout.addLayout(hbox)

        # Add a spacer item at the bottom to push the buttons to the center vertically
        spacer_bottom = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        main_layout.addItem(spacer_bottom)

        self.central_widget.setLayout(main_layout)

    def on_button1_click(self):
        # Open file dialog to select an image
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "Image Files (*.jpg *.jpeg *.png)", options=options)
        if file_path:
            try:
                # Run YOLOv8 model on the selected image
                model = YOLO("yoloD1E100.pt")
                results = model.predict(source=file_path, conf=0.25, save=True)

                # Get the latest predict directory (assuming it's the highest number)
                predict_dirs = glob.glob("runs/detect/predict*")
                latest_predict_dir = max(predict_dirs, key=os.path.getctime)
                output_image_path = os.path.join(latest_predict_dir, os.path.basename(file_path))

                # Display the output image
                self.display_output_image(output_image_path)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def display_output_image(self, image_path):
        # Display the output image in a message box or any other widget
        msg = QMessageBox(self)
        msg.setWindowTitle("YOLOv8 Output Image")
        pixmap = QPixmap(image_path)
        msg.setIconPixmap(pixmap)
        msg.exec_()

    def on_button2_click(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a Video", "", "Video Files (*.mp4 *.avi)", options=options)
        if file_path:
            try:
                self.progress_bar = QProgressBar(self)
                self.progress_bar.setGeometry(30, 40, 200, 25)
                self.progress_bar.setMaximum(100)
                self.progress_bar.setValue(0)
                self.progress_bar.show()

                self.video_thread = VideoProcessingThread(video_path=file_path, model=YOLO("yoloD1E100.pt"))
                self.video_thread.progress_updated.connect(self.update_progress)
                self.video_thread.processing_finished.connect(self.processing_finished)
                self.video_thread.start()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def update_progress(self, progress):
        self.progress_bar.setValue(progress)

    def processing_finished(self, output_path):
        self.progress_bar.hide()
        QMessageBox.information(self, "Processing Finished", "Video processing has completed!")
        self.open_video(output_path)

    def open_video(self, video_path):
        if sys.platform == "win32":
            os.startfile(video_path)
        elif sys.platform == "darwin":
            subprocess.call(('open', video_path))
        else:
            subprocess.call(('xdg-open', video_path))

    def start_webcam(self):
        if not self.capture:
            try:
                self.model = YOLO("yoloD1E100.pt")  # Initialize YOLO model
                self.capture = cv2.VideoCapture(0)  # Open webcam

                # Start timer to read frames
                self.timer.timeout.connect(self.display_webcam)
                self.timer.start(30)  # Update frame every 30 milliseconds
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to initialize YOLOv5 model: {str(e)}")

    def stop_webcam(self):
        if self.capture:
            self.timer.stop()
            self.capture.release()
            self.capture = None

    def closeEvent(self, event):
        # Stop webcam and release resources when closing the window
        self.stop_webcam()
        event.accept()

    def display_webcam(self):
        ret, frame = self.capture.read()  # Read frame from webcam
        if ret:
            # Convert frame to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform inference with YOLO model
            results = self.model(source=0, show=True)  # Adjust size if needed
            
            # Draw bounding boxes on the frame
            for box in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = box.tolist()
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f"{self.model.names[int(cls)]}: {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Convert frame to QImage
            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Display QImage in QLabel
            self.label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        # Stop webcam and release resources when closing the window
        self.stop_webcam()
        event.accept()
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())