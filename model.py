import os
from typing import Type
import cv2
import mediapipe as mp
import numpy as np
from keras.layers import Dropout
from mediapipe.python.solutions.holistic import Holistic
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from constants import ACTIONS, NO_SEQUENCES, DATA_PATH, SEQUENCE_LENGTH

class Model:
    """Builds and saves a model using camera frames for gesture recognition."""
    
    def __init__(self, model_name: str) -> None:
        """
        Initializes the model with necessary configurations.
        Args:
            model_name (str): Name for the saved model file.
        """
        # MediaPipe's holistic model for hand tracking and drawing utils for visualing landmarks
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.model_name = model_name
        
        
    def predict_hand_landmarks(self, frame: np.ndarray, model: Holistic) -> Type:
        """
        Predicts hand landmarks using the holistic model.
            Args:
                frame (np.ndarray): Frame captured from the camera.
                model (Holistic): Holistic model for prediction.
            Returns:
                Type: Prediction results.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = model.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return results
    
    
    def create_directory(self) -> None:
        """Creates directories for storing sequences of action frames.
        """
        for action in ACTIONS:
            for sequence in range(NO_SEQUENCES):
                try:
                    os.makedirs(os.path.join("data", DATA_PATH, action, str(sequence)))
                except Exception as e:
                    print(f"Error creating directory: {e}")
                    
                    
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
    
    
    def save_frame(self, results: Type, action: str, sequence: int, frame_num: int) -> None:
        """ Saves the frame's landmark data

        Args:
            results (Type): Predicted hand landmarks
            action (str): Action name.
            sequence (int): Sequence number.
            frame_num (int): Frame number within the sequence.
        """
        landmarks = self.save_hand_landmarks(results)
        path = os.path.join("data", DATA_PATH, action, str(sequence), str(frame_num))
        np.save(path, landmarks)
        
    def create_dataset(self) -> None:
        """Captures and saves sequences of action frames using the camera.
        """
        self.create_directory()
        
        with self.mp_holistic.Holistic(min_detection_condifence=0.5, min_tracking_confidence=0.5) as holistic:
            for action in ACTIONS:
                for sequence in range(NO_SEQUENCES):
                    cv2.waitKey(1000)
                    print(f"Starting action: {action}, Sequence: {sequence}")
                    for frame_num in range(SEQUENCE_LENGTH):
                        ret, frame = self.cap.read()
                        
                        if not ret:
                            print("Error with camera.")
                            break
                        
                        results = self.hand_prediction(frame, holistic)
                        self.draw_hand_landmarks(frame, results)
                        self.save_frame(results, action, sequence, frame_num)
                        
                        cv2.imshow("Creating dataset for hand gesture recognition", frame)
                        
                        if cv2.waitKey(10) & 0xFF = ord('q'):
                            break
            self.cap.release()
            cv2.destroyAllWindows()        