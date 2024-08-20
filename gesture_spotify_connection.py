from queue import Queue
from threading import Thread

from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtGui import QImage

from gesture_recognition import gesture_recognition
from spotify_connector import spotify_api

class gesture_spotify_connection(QThread):
    """ Class that integrates gesture recognition with Spotify controls.

    Args:
        QThread (_type_): _description_
    """
    
    # Signals to update UI with detected gesture, confidence, and processed image
    change_gesture_name = pyqtSignal(str)
    change_confidence = pyqtSignal(str)
    change_image = pyqtSignal(QImage)
    
    def __init__(self, 
                 client_id: str = "", 
                 client_secret: str = "", 
                 token: str = "") -> None:
        """
        Connection constructor.
        
        Args:
            client_id (str): the unique identifier of user app
            client_secret (str): key used to authorize
            token(str): OAuth token for Spotify API
        """
        super().__init__
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = token
        self.spotify = spotify_api(token=self.token)
        # Gesture model file
        self.hand_detetction_model = gesture_recognition("model/model.h5")
        
    
    def start_detection(self) -> None:
        """
        Function creates threads - detect gestures and make action based on gesture name.
        """
        gesture_queue = Queue()
        
        # Create a model action thread, pushing detected gestures into queue
        detection_thread = Thread(
            target=self.hand_detetction_model.detect_gestures, 
            args=(gesture_queue, self.change_image,), 
            daemon = True)
        
        # Listens to the queue for gestures and performs the appropriate action on Spotify
        spotify_action_thread = Thread(
            target = self.gesture_action, 
            args=(gesture_queue,), 
            daemon = True)
        
        detection_thread.start()
        spotify_action_thread.start()
    
    def gesture_action(self, gesture_queue: Queue[str]) -> None:
        """
        Function makes action based on gesture name from input_gesture queue

        Args:
            input_gesture (Queue[str]): queue with information about detected gesture
        """
        while True:
            gesture_info = gesture_queue.get()
            gesture = gesture_info.get('gesture')
            gesture = gesture_info.get('confidence')
            
            if gesture:
                self.change_gesture_name.emit(gesture)
                self.change_confidence.emit(confidence)
                
                # Perform action corresponding to the detected gesture
                gesture_actions = {
                    "next": self.spotify.skip_to_next,
                    "prev": self.spotify.skip_to_previous,
                    "love": self.spotify.fav_track,
                    "louder": self.spotify.volume_up,
                    "quieter": self.spotify.volume_down,
                    "play_pause": self.spotify.change_playing_status
                }
                
                action = gesture_actions.get(gesture)
                if action:
                    action()