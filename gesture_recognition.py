from queue import Queue
from typing import Type

import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage
from keras.models import load_model
from mediapipe.python.solutions.holistic import Holistic

from constants import ACTIONS, THRESHOLD

class gesture_recognition:
    """ Class for recognizing gestures from camera images using a pre-trained MediaPipe Holistic
    model.
    """
    change_image = pyqtSignal(QImage)
    
    def __init__(self, model_name: str) -> None:
        """HandDetectionModel constructor to set up the camera and MediaPipe components
        Arg:
            model_name(str)": file name to load the model"""
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.model_name = model_name
        
        self.gesture = None
        self.confidence = None
        
    def generate_hand_prediction(self, frame: np.ndarray, model: Holistic) -> Type:
        """Function generate prediction based on holistic model
        Args:
            frame (np.ndarray): frame from the camera
            model (Holistic): holistic model used for prediction"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = model.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return results
    
    def draw_hand_landmarks(self, image: np.ndarray, results: Type) -> None:
        """Draws visual representations of the detected hand landmarks on the image frame.
        
        Args:
            image (np.ndarray): Frame from the camera.
            results (Type): Predicted hand landmarks and connections.
        """
        landmarks_left_hand = self.mp_drawing.DrawingSpec(
            color=(255,20,147), thickness=2, circle_radius=2
        )
        landmarks_right_hand = self.mp_drawing.DrawingSpec(
            color=(0,0,255), thickness=2, circle_radius=2
        )
        connection_line = self.mp_drawing.DrawingSpec(
            thickness=4, color=(90, 90, 90)
        )
        
        self.mp_drawing.draw_landmarks(
            image, 
            results.left_hand_landmarks, 
            self.mp_holistic.HAND_CONNECTIONS, 
            landmarks_left_hand, 
            connection_line
        )
        
        self.mp_drawing.draw_landmarks(
            image, 
            results.right_hand_landmarks, 
            self.mp_holistic.HAND_CONNECTIONS, 
            landmarks_right_hand, 
            connection_line
        )
    
    def save_hand_landmarks(self, results: Type) -> np.ndarray:
        """Function creates array with coordinates of hand landmarks.

        Args:
            results (Type): predicted hand

        Returns:
            np.ndarray: an array object with coordinates of landmakrs
        """
        left_hand_points = right_handpoints = np.zeros(63)
        
        if results.left_hand_landmarks:
            left_hand_points = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
            
        if results.right_hand_landmarks:
            right_hand_points = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
            
        return np.concatenate([left_hand_points, right_handpoints])
    
    def convert_image(self, image: np.ndarray) -> QImage:
        """Function convers np.ndarray from camera to an QImage object to display it.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_image.shape
        converted_image = QImage(rgb_image.data, w, h, QImage.Format_RGB888)
        scaled_image = converted_image.scaled(600, 600, Qt.KeepAspectRatio)
        return scaled_image
    
    def detect_gestures(self, output_gesture: Queue[str], change_image) -> None:
        """Function detects gestures based on image from camera and adds detected gesture's name to the queue

        Args:
            output_gesture (Queue[str]): queue to save detected gesture's information
            change_image (_type_): pass
        """
        landmarks_from_frame = []
        
        model = load_model(self.model_name)
        
        # Set up MediaPipe Holistic Model for hand tracking and pose detection
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # Asure camera is active
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Camera issue detected. Exiting the application...")
                    break
                
                # Generate hand prediction
                results = self.generate_hand_prediction(frame, holistic)
                
                # Draw hands landmarks and update the displayed image with processed frame
                self.draw_hand_landmarks(frame, results)
                change_image.emit(self.convert_image(frame))
                
                # Save landmarks and add them to landmark list
                landmarks = self.save_hand_landmarks(results)
                landmarks_from_frame.append(landmarks)
                
                # Create a sequence for gesture recognition. Keep last 30 frames of landmarks in the list
                landmarks_from_frame = landmarks_from_frame[-30:]
                
                if len(landmarks_from_frame) == 30:
                    transformed_data = np.expand_dims(landmarks_from_frame, axis=0)
                    prediction = model.predict(transformed_data)[0]
                    
                    max_prediction_index = np.argmax(prediction)
                    
                    if prediction[max_prediction_index] > THRESHOLD:
                        self.confidence = prediction[max_prediction_index]
                        self.gesture = ACTIONS[max_prediction_index]
                        gesture_info = {
                            "gesture": self.gesture,
                            "confidence": str(round(self.confidence*100, 2))
                        }
                        output_gesture.put(gesture_info)
                        landmarks_from_frame = []
                        
                    cv2.waitKey(10)
                    
                self.cap.release()
                cv2.destroyAllWindows()
                
            self.cap.release()
            cv2.destroyAllWindows()