# Hand-Gesture-Drone-Control


Overview
This project presents an intuitive drone control system leveraging gesture and voice recognition to facilitate seamless navigation and dynamic speed control. Designed with versatility and precision in mind, it enables users to control one or two drones simultaneously using dual-hand gestures, significantly enhancing usability in multi-drone environments.

Features

🎮 Gesture-Based Navigation
Control drone movement with natural hand gestures using real-time tracking via MediaPipe.

🎤 Voice-Controlled Commands
Execute predefined drone commands (e.g., takeoff, land, hover) using voice recognition for hands-free operation.

✋✋ Dual-Hand Gesture Support
Independently control two drones using both hands, enabling parallel navigation tasks and advanced coordination.

🧠 ANN-Based Gesture Recognition
An Artificial Neural Network (ANN) trained on MediaPipe landmarks ensures accurate interpretation of complex gestures, minimizing errors in command execution.

Technologies Used

MediaPipe – Hand landmark detection

TensorFlow/Keras – ANN training for gesture recognition

SpeechRecognition / PyAudio – Voice command processing

Drone SDK (e.g., DJI/Tello SDK) – Interface with drones (replace with actual SDK used)

How It Works

Gesture Detection: MediaPipe captures 3D hand landmarks from webcam input.

Gesture Classification: A trained ANN processes the landmarks to identify gesture commands.

Voice Recognition: Voice commands are parsed using speech recognition tools.

Drone Control: Identified commands are translated into drone SDK instructions for movement or action.
