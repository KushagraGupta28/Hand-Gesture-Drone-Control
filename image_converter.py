import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import pandas as pd  # type: ignore
import os

# MEDIAPIPE
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


gesture_folders = {  
    'hand_gesture_dataset/flip': 6,    
    'hand_gesture_dataset/land': 7
}

# COLUMNS
columns = ['gesture_id'] + [f"x{i}_landmark" for i in range(21)] + [f"y{i}_landmark" for i in range(21)]
df = pd.DataFrame(columns=columns)

# LANDMARKS
def extract_hand_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    results = hands.process(image_rgb)

    
    if not results.multi_hand_landmarks:
        return None

    
    landmarks = results.multi_hand_landmarks[0]
    coords = []
    for landmark in landmarks.landmark:
        coords.append(landmark.x*640)  # x coordinate
        coords.append(landmark.y*480)  # y coordinate

    return coords



for folder, gesture_id in gesture_folders.items():
    image_files = os.listdir(folder)

    for image_file in image_files:
        image_path = os.path.join(folder, image_file)

        
        coords = extract_hand_landmarks(image_path)

        if coords:
            
            row = [gesture_id] + coords  
            # APPENDING OF ROW
            df.loc[len(df)] = row
            print(f"Processed {image_file} ")
        else:
            print(f"ERROR IN THE IMAGE")

csv_file = 'hand_gesture_new_landmarks.csv'
df.to_csv(csv_file, index=False)

print("TASK COMPLETED")
