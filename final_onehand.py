import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import time
import numpy as np  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore

# GESTURES
class_labels = ["UP", "DOWN", "LEFT", "RIGHT", "BACK", "FRONT","FLIP","LAND"]


model = load_model("my_model_gesture_prediction.keras")

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1) 
draw = mp.solutions.drawing_utils

video = cv2.VideoCapture(0)
confidence_threshold = 0.99
c_t = 0
p_t = 0

while True:
    ret, image123 = video.read()
    image = cv2.flip(image123, 1)
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    result = hands.process(converted)

    c_t = time.time()
    fps = 1 / (c_t - p_t)
    p_t = c_t

    # FPS
    cv2.putText(image, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
    # Check if hands are detected
    if result.multi_hand_landmarks:
        
        hand_landmarks = result.multi_hand_landmarks[0]
        frame_landmarks = []  # To hold all x, y pixel coordinates for the current hand

        
        for landmark_id, lm in enumerate(hand_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            frame_landmarks.extend([cx, cy])

        
        
        input_data = np.array([frame_landmarks])
        prediction = model.predict(input_data)
        max_prob = np.max(prediction)
        y_pred_class = np.argmax(prediction, axis=1)
        predicted_class_index = y_pred_class[0]
        predicted_class_label = class_labels[predicted_class_index]



        if max_prob < confidence_threshold:
            predicted_class_label = "None" 
        else:
            predicted_class_label = class_labels[predicted_class_index]

            
        

        
        cv2.putText(image, f"DRONE : {predicted_class_label}",(100,100),cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)

        
        draw.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS, 
                            draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                            draw.DrawingSpec(color=(255, 0, 0), thickness=2))

    # Display the output
    cv2.imshow("SCREEN", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
