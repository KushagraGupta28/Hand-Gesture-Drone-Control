import speech_recognition as sr  # type: ignore

def main():
    """
    The main function to continuously listen for voice commands and process them.
    Supported commands: 'up', 'down', 'left', 'right', 'front', 'back'.
    """
    # Create a Recognizer object
    recognizer = sr.Recognizer()
    
    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        # Dynamically adjust sensitivity to ambient noise
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("Listening... You can start speaking now.")
        
        while True:  # Infinite loop to continuously listen
            try:
                # Listen for audio from the microphone with increased timeout and phrase time limit
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)  # Adjusted timeout and phrase time limit

                # Convert the captured audio into text using Google Web Speech API
                command = recognizer.recognize_google(audio).lower()  # Convert to lowercase for uniformity
                
                # Check if the recognized command is in the list of supported commands
                if command in ["up", "down", "left", "right", "front", "back"]:
                    print(f"{command}")  # Output the command as you speak
                else:
                    print("Unrecognized command.")  # Output for unrecognized commands

            except sr.UnknownValueError:
                # Handle the case when speech is not understood
                pass  # Ignore and continue listening
            except sr.RequestError:
                # Handle API connection errors
                print("Error with the Google Web Speech API.")
                break  # Exit on API error
            except sr.WaitTimeoutError:
                # Handle the timeout error explicitly to prevent program termination
                continue  # Ignore timeout and continue listening
            except Exception as e:
                # Handle any other unexpected errors
                print(f"An unexpected error occurred: {e}")
                break  # Exit on unexpected error

if __name__ == "__main__":
    """
    The main entry point of the program.
    Continuously listens for commands until the program is terminated.
    """
    try:
        main()  # Call the main function for continuous listening
    except KeyboardInterrupt:
        print("\nProgram terminated by user. Goodbye!")
