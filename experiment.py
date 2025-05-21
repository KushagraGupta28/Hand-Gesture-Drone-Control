import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import time
import numpy as np  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from threading import Thread

# GESTURES
class_labels = ["UP", "DOWN", "LEFT", "RIGHT", "BACK", "FRONT", "FLIP", "LAND"]

model = load_model("my_model_gesture_prediction.keras")

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils

# Global variables for threading
frame = None
stop_thread = False
fps = 0
confidence_threshold = 0.99
predicted_class_label = "None"

def video_capture_thread():
    global frame, stop_thread
    video = cv2.VideoCapture(0)
    while not stop_thread:
        ret, frame_local = video.read()
        if ret:
            frame = cv2.flip(frame_local, 1)
    video.release()

# Start the video capture thread
capture_thread = Thread(target=video_capture_thread)
capture_thread.start()

prev_time = 0

while True:
    if frame is None:
        continue

    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(converted)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Check if hands are detected
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        frame_landmarks = []  # To hold all x, y pixel coordinates for the current hand

        for lm in hand_landmarks.landmark:
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            frame_landmarks.extend([cx, cy])

        # Predict gesture
        input_data = np.array([frame_landmarks])
        prediction = model.predict(input_data)
        max_prob = np.max(prediction)
        y_pred_class = np.argmax(prediction, axis=1)
        predicted_class_index = y_pred_class[0]

        if max_prob < confidence_threshold:
            predicted_class_label = "None"
        else:
            predicted_class_label = class_labels[predicted_class_index]

        # Display the prediction
        cv2.putText(frame, f"DRONE : {predicted_class_label}", (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS,
                            draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                            draw.DrawingSpec(color=(255, 0, 0), thickness=2))

    # Display FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Display the output
    cv2.imshow("SCREEN", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_thread = True
        break

capture_thread.join()
cv2.destroyAllWindows()
