from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QWidget

from gesture_spotify_connection import gesture_spotify_connection
from pyui.detection_window import Ui_DetectionWindow

class detection_window(QDialog, Ui_DetectionWindow):
    """The class generates detetction window of GUI.

    Args:
        QDialog (_type_): _description_
        Ui_DetectionWindow (_type_): _description_
    """
    def __init__(self) -> None:
        super().__init__()
        self.token = ""
        self.setupUi(self)
        self.connection = gesture_spotify_connection(self)
        self._connect_buttons()
        
    def _connect_buttons(self) -> None:
        self.pushButton_2.clicked.connect(self.connection_action)
    
    def connection_action(self) -> None:
        self.connection.change_gesture_name.connect(self.update_gesture_name)
        self.connection.change_gesture_name.connect(self.update_gesture_name)
        self.connection.change_image.connect(self.update_image)
        self.connection.token = self.token
        self.connection.started.connect(self.connection.start_detection)
        self.connection.start()
    
    def destroy_thread(self):
        self.connection.quit()
        self.connection = None
    
    @pyqtSlot(str)
    def update_confidence(self, confidence: str) -> None:
        self.detection_confidence.setText(f"Confidence: {confidence}%")
        
    @pyqtSlot(str)
    def update_gesture_name(self, confidence: str) -> None:
        self.detected_gesture_name.setText(f"Gesture name: {gesture_name}%")
        
    @pyqtSlot(str)
    def update_image(self, image: QImage) -> None:
        self.detection_camera.setPixmap(QPixmap.fromImage(image))