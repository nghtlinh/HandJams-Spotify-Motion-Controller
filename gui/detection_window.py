from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog

from core.gesture_spotify_connection import gesture_spotify_connection
from pyui.detection_window import Ui_DetectionWindow
from queue import Queue

class detection_window(QDialog, Ui_DetectionWindow):
    """The class generates detetction window of GUI.

    Args:
        QDialog (_type_): _description_
        Ui_DetectionWindow (_type_): _description_
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.token = ""
        self.connection = gesture_spotify_connection(self)
        self._connect_buttons()
        self.queue = Queue()
        
    def _connect_buttons(self) -> None:
        self.pushButton_2.clicked.connect(self.connection_action)
    
    def connection_action(self) -> None:
        self.connection.change_gesture_name.connect(self.update_gesture_name)
        self.connection.change_confidence.connect(self.update_confidence)
        self.connection.change_image.connect(self.update_image)
        
        self.connection.add_gesture.connect(self.add_gesture)
        self.connection.token = self.token
        self.connection.start()
    
    @pyqtSlot(str)
    def add_gesture(self, gesture_name: str) -> None:
        self.q.put(gesture_name)
    
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