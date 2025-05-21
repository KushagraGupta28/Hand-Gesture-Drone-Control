import speech_recognition as sr
import google.generativeai as genai
import asyncio
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time

# Setup for speech recognition and GenAI model
recognizer = sr.Recognizer()
genai.configure(api_key="AIzaSyAMMQRp2b_L0ZJ5HS_nqwnPvuiYTBJhKD4")

generation_config = {
    'temperature': 0.7,
    'top_p': 0.95,
    'top_k': 40,
    'max_output_tokens': 200,
    'response_mime_type': 'text/plain'
}

my_model = genai.GenerativeModel(
    model_name='gemini-2.0-flash-exp',
    generation_config=generation_config,
    system_instruction="You are a answer generator. you only reply in true or false. use only true or false nothing else. your name is drone if i ask this return true"
)

history = []

# Hand gesture model and mediapipe initialization
class_labels = ["UP", "DOWN", "LEFT", "RIGHT", "BACK", "FRONT", "FLIP", "LAND"]
model = load_model("my_model_gesture_prediction.keras")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

confidence_threshold = 0.6
c_t = 0
p_t = 0

# Asynchronous Speech recognition function
async def speech_recognition():
    with sr.Microphone() as source:
        print("bot: HELLO, HOW CAN I HELP YOU? (Press Ctrl+C to stop)")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise

        while True:  # Infinite loop to keep listening
            try:
                print("Listening...")  # Indicate that it's listening
                audio = recognizer.listen(source)  # Listen for the first phrase

                # Recognize speech using Google's speech recognition API
                user_input = recognizer.recognize_google(audio)
                print(f"USER: {user_input}")  # Print the recognized question

                # Start the chat session and send the recognized user input
                chat_session = my_model.start_chat(history=history)
                response = chat_session.send_message(user_input)
                model_response = response.text

                print(f"bot: {model_response}")  # Print the response from the bot

            except sr.UnknownValueError:
                print("Sorry, I could not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
            except KeyboardInterrupt:
                print("\nProgram stopped by user.")
                break  # Exit the loop when Ctrl+C is pressed

# Asynchronous Gesture recognition function
async def gesture_recognition():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mediapipe gesture recognition code
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        frame_height, frame_width, _ = frame.shape

        # Draw dividing line and drone labels
        cv2.line(frame, (frame_width // 2, 0), (frame_width // 2, frame_height), (0, 0, 255), 2)
        cv2.putText(frame, "DRONE 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, "DRONE 2", (frame_width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Calculate FPS
        c_t = time.time()
        fps = 1 / (c_t - p_t)
        p_t = c_t
        cv2.putText(frame, f"FPS: {int(fps)}", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                frame_landmarks = []
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * frame_width), int(lm.y * frame_height)
                    frame_landmarks.extend([cx, cy])

                # Convert to NumPy array and predict gesture
                input_data = np.array([frame_landmarks])
                prediction = model.predict(input_data, verbose=0)
                max_prob = np.max(prediction)
                y_pred_class = np.argmax(prediction, axis=1)
                predicted_class_label = class_labels[y_pred_class[0]] if max_prob >= confidence_threshold else "None"

                # Determine which drone is controlled
                drone = "DRONE 1" if wrist_x < 0.5 else "DRONE 2"

                # Display gesture prediction
                x_pos = 50 if drone == "DRONE 1" else frame_width - 200
                cv2.putText(frame, f"{drone}: {predicted_class_label}", 
                            (x_pos, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Draw landmarks and connections
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

        # Display the frame
        cv2.imshow("Drone Control", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function that runs the asynchronous tasks
async def main():
    # Run both functions concurrently
    await asyncio.gather(speech_recognition(), gesture_recognition())

# Run the main function
asyncio.run(main())
