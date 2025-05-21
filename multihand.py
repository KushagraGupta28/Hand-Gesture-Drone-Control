import cv2 # type: ignore
import mediapipe as mp # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import time

# model+gestures
class_labels = ["UP", "DOWN", "LEFT", "RIGHT", "BACK", "FRONT", "FLIP", "LAND"]
model = load_model("my_model_gesture_prediction.keras")

# mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)
confidence_threshold = 0.6
c_t = 0
p_t = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    
    frame_height, frame_width, _ = frame.shape

    # dividing line
    cv2.line(frame, (frame_width // 2, 0), (frame_width // 2, frame_height), (0, 0, 255), 2)
    cv2.putText(frame, "DRONE 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, "DRONE 2", (frame_width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    # Calculate FPS
    c_t = time.time()
    fps = 1 / (c_t - p_t)
    p_t = c_t
    cv2.putText(frame, f"FPS: {int(fps)}", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # x coordinate to determine which half of screen 
            wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x

            
            frame_landmarks = []
            for lm in hand_landmarks.landmark:
                cx, cy = int(lm.x * frame_width), int(lm.y * frame_height)
                frame_landmarks.extend([cx, cy])
            

            
            input_data = np.array([frame_landmarks])
            prediction = model.predict(input_data, verbose=0)
            max_prob = np.max(prediction)
            y_pred_class = np.argmax(prediction, axis=1)
            if max_prob >= confidence_threshold:
                predicted_class_label = class_labels[y_pred_class[0]]  
            else:
                "None"

            # drone determination
            drone = "DRONE 1" if wrist_x < 0.5 else "DRONE 2"

            
            x_pos = 50 if drone == "DRONE 1" else frame_width - 200
            cv2.putText(frame, f"{predicted_class_label}",(x_pos, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # draw
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

    
    cv2.imshow("Drone Control", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






# DOUBTS
# 1. HOW TO SET THIS THRESHOLD VALUE
# 2. DATASET MAI KITNI IMAGES FOR A CLASS IN MEDIAPINE
# 3. IF ONLY CNN THNE KITNI IMAGES , AND UNKO AUGMENT KARKE KITNI BANANI CHAHIYE
# 4. WHETHER AUGMENTATION BE HELPFULL IN MEDIAPIPE COORDINATES WALI CHEEZ
# 5. MODEL MAI NUMBER OF LAYERS AND UNITS KAISE DETEMINE KARNI CHAHIYE
# 6. THREADING MAI ERROR KYU ARAHA  