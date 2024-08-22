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
        # MediaPipe's holistic model for hand tracking and drawing utils for visualizing landmarks
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
        """Creates directories for storing sequences of action frames."""
        for action in ACTIONS:
            for sequence in range(NO_SEQUENCES):
                try:
                    os.makedirs(os.path.join("data", DATA_PATH, action, str(sequence)))
                except Exception as e:
                    print(f"Error creating directory: {e}")

    def draw_hand_landmarks(self, image: np.ndarray, results: Type) -> None:
        """Draws visual representations of the detected hand landmarks on the image frame."""
        landmarks_left_hand = self.mp_drawing.DrawingSpec(
            color=(255, 20, 147), thickness=2, circle_radius=2
        )
        landmarks_right_hand = self.mp_drawing.DrawingSpec(
            color=(0, 0, 255), thickness=2, circle_radius=2
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
        """Creates array with coordinates of hand landmarks."""
        left_hand_points = right_hand_points = np.zeros(63)
        if results.left_hand_landmarks:
            left_hand_points = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks:
            right_hand_points = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
        return np.concatenate([left_hand_points, right_hand_points])

    def save_frame(self, results: Type, action: str, sequence: int, frame_num: int) -> None:
        """Saves the frame's landmark data."""
        landmarks = self.save_hand_landmarks(results)
        path = os.path.join("data", DATA_PATH, action, str(sequence), str(frame_num))
        np.save(path, landmarks)

    def create_dataset(self) -> None:
        """Captures and saves sequences of action frames using the camera."""
        self.create_directory()
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for action in ACTIONS:
                for sequence in range(NO_SEQUENCES):
                    cv2.waitKey(1000)
                    print(f"Starting action: {action}, Sequence: {sequence}")
                    for frame_num in range(SEQUENCE_LENGTH):
                        ret, frame = self.cap.read()
                        if not ret:
                            print("Error with camera.")
                            break
                        results = self.predict_hand_landmarks(frame, holistic)
                        self.draw_hand_landmarks(frame, results)
                        self.save_frame(results, action, sequence, frame_num)
                        cv2.imshow("Creating dataset for hand gesture recognition", frame)
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
            self.cap.release()
            cv2.destroyAllWindows()

    def prepare_data(self) -> tuple:
        """Prepares the dataset by loading pre-saved sequences of landmark data from the '.npy' file."""
        labels = {action: i for i, action in enumerate(ACTIONS)}
        data_sequences, data_labels = [], []
        for action in ACTIONS:
            for sequence in range(NO_SEQUENCES):
                frames_in_sequence = []
                for frame_num in range(SEQUENCE_LENGTH):
                    path = os.path.join("data", DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                    loaded_frame = np.load(path)
                    frames_in_sequence.append(loaded_frame)
                data_sequences.append(frames_in_sequence)
                data_labels.append(labels[action])
        data_sequences = np.array(data_sequences)
        data_labels = to_categorical(data_labels, dtype="uint8")
        return data_sequences, data_labels

    def create_model(self) -> None:
        """Trains and saves the gesture recognition model."""
        sequences, labels = self.prepare_data()
        sequences_train, sequences_test, labels_train, labels_test = train_test_split(
            sequences, labels, test_size=0.3, random_state=42
        )
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 126)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=True, activation='relu'))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(len(ACTIONS), activation='softmax'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.fit(sequences_train, labels_train, epochs=100, batch_size=32, validation_split=0.1)
        model.save(f'model/{self.model_name}.h5')
        del model

    def run(self) -> None:
        """Runs the full workflow for data collection and model creation."""
        print("Starting data collection...")
        self.create_dataset()
        print("Data collection complete. Starting model training...")
        self.create_model()
        print("Model training complete and saved.")

# Usage
if __name__ == "__main__":
    model_instance = Model(model_name="gesture_recognition_model")
    model_instance.run()
