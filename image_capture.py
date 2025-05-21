import cv2 # type: ignore
import time
import os

c_t= time.time()


gesture = 'land'
images_per_gesture = 1000
capture_interval = 0.2  

# FOLDER CHECK
main_folder = 'hand_gesture_dataset'
if not os.path.exists(main_folder):
    os.mkdir(main_folder)

# SUBFOLDER 
gesture_folder = os.path.join(main_folder, gesture)
if not os.path.exists(gesture_folder):
    os.mkdir(gesture_folder)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()


def capture_gesture_images(gesture, images_per_gesture):
    count = 0
    last_capture_time = time.time() + 10 
    
    
    while count < images_per_gesture:
        ret, frame = cap.read()
        if not ret:
            print("FAILED")
            break

        current_time = time.time()


        timep= current_time - c_t
        cv2.putText(frame, f"Time: {timep:.2f}s", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        
        
        cv2.imshow("Gesture Capture", frame)
        
        if current_time - last_capture_time >= capture_interval:
           
            #timep = current_time - last_capture_time
            #cv2.putText(frame, f"Time: {timep:.2f}s", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
            
            
            
            
            # Save the image
            image_name = f"{gesture}_{count + 1}.jpg"
            image_path = os.path.join(gesture_folder, image_name)
            cv2.imwrite(image_path, frame)
            
            # TIME UPDATE 
            last_capture_time = current_time
            count += 1
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
        cv2.imshow("Gesture Capture", frame)
    
    print(f"Finished capturing {images_per_gesture} images for {gesture}")


capture_gesture_images(gesture, images_per_gesture)


cap.release()
cv2.destroyAllWindows()

print("TASK COMPLETED")
