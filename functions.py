import cv2
import numpy as np
import json
from matplotlib import pyplot as plt
import time
import glob
import mediapipe as mp

# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import LSTM, Dense, GRU, Dropout

# from tensorflow.keras.callbacks import TensorBoard

# def create_model(model_path):
    # model = Sequential()
    # model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(25,1662)))
    # model.add(LSTM(128, return_sequences=True, recurrent_dropout = 0.2, activation='relu'))
    # model.add(LSTM(64, return_sequences=False, recurrent_dropout = 0.2, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(10, activation='softmax'))

    # model.load_weights(model_path)
    # return model

class Predictor:
    def __init__(self, model_path):
        # self.model = create_model(model_path)
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils 
        self.actions = ['drink', 'computer', 'before', 'go', 'who']
        self.threshold = 0.8
        self.counts = []
    
    def mediapipe_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = self.model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results
    
    def draw_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION) # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS) # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
    
    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    def most_frequent(List):
        return max(set(List), key = List.count)

    def predict(self, video_path):
        cap = cv2.VideoCapture(video_path)

        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while(True):

                ret, frame = cap.read()
                if ret == False:
                    break

                cv2.imshow('frame', frame)

                if cv2.waitKey(10) & 0xff == ord('q'):
                    break
                # Make detections
        #         image, results = self.mediapipe_detection(frame, holistic)
        #         # print(results)
        #
        #         # Draw landmarks
        #         self.draw_styled_landmarks(image, results)
        #
        #         # 2. Prediction logic
        #         keypoints = self.extract_keypoints(results)
        # #         sequence.insert(0,keypoints)
        # #         sequence = sequence[:30]
        #         sequence.append(keypoints)
        #         sequence = sequence[-100:]
        #
        #         if len(sequence) == 100:
        #             res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
        #
        #
        #         #3. Viz logic
        #             if res[np.argmax(res)] > self.threshold:
        #                 self.counts.append(self.actions[np.argmax(res)])
        #                 if len(sentence) > 0:
        #                     if self.actions[np.argmax(res)] != sentence[-1]:
        #                         sentence.append(self.actions[np.argmax(res)])
        #                 else:
        #                     sentence.append(self.actions[np.argmax(res)])
        #
        #             if len(sentence) > 5:
        #                 sentence = sentence[-5:]


        return 'arm'