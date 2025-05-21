import speech_recognition as sr # type: ignore
import google.generativeai as genai # type: ignore

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


with sr.Microphone() as source:
    print("bot: HELLO, HOW CAN I HELP YOU? (Press Ctrl+C to stop)")

    recognizer = sr.Recognizer()
    recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise

    while True:  
        try:
            print("Listening - ") 
            audio = recognizer.listen(source)  

            
            user_input = recognizer.recognize_google(audio)
            print(f"USER: {user_input}")

            
            chat_session = my_model.start_chat(history=history)
            response = chat_session.send_message(user_input)
            model_response = response.text

            print(f"bot: {model_response}")

        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        except KeyboardInterrupt:
            print("\nProgram stopped by user.")
            break 