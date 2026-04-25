import cv2
import webbrowser
import numpy as np
import pygetwindow as gw
import pygame
import os
import random
from deepface import DeepFace
from gtts import gTTS

# Load OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize Pygame for Sound Playback
pygame.mixer.init()

# Dictionary for Mood-Based Songs (Fixed list issue)
mood_songs = {
    "sad": [
        "https://youtu.be/sFMRqxCexDk?si=eReG8Cm8NteAMhAA",
        "https://youtu.be/sVRwZEkXepg?si=42qBdoPB_Qr7tpVw",
        "https://youtu.be/zSDgwvQoSHg?si=aFy0GxFuw4Jet8j7"
    ],
    "happy": [
        "https://youtu.be/_hRnWA0A20E?si=VsalfJZj0Ib4yDj6",
        "https://youtu.be/8uJ-wOljP_s?si=zAlBEjzjxFnm17OI",
        "https://youtu.be/igQCv_Y33NI?si=PlsJJaOwQqL8EnJZ"
    ],
    "angry": [
        "https://youtu.be/0gK1SHLtaOo?si=vYlmOjuIYQli5X0z",
        "https://youtu.be/zqGW6x_5N0k?si=NlcGn8owxntFTB34",
        "https://youtu.be/n_FCrCQ6-bA?si=3DA-CT9Cw5Okj1IW"
    ],
    "surprise": ["https://youtu.be/II2EO3Nw4m0?si=MQzDryytxJXbCzw-"],
    "neutral": [
        "https://youtu.be/AMuRRXCuy-4?si=_9T1r8pGyjTcqB6n",
        "https://youtu.be/HSvMZEyd7uc?si=EktzXXC7pm5yEAHL"
    ]
}

# Variables to Track Mood Changes
last_mood = None
song_played = set()  # Store played moods
frame_count = 0  # Counter to reduce CPU load

# Function to Make Jarvis Speak
def jarvis_speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save("jarvis_voice.mp3")

        pygame.mixer.music.load("jarvis_voice.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
        pygame.mixer.music.unload()
        os.remove("jarvis_voice.mp3")
    except Exception as e:
        print(f"Voice Error: {e}")

# Function to Check if Jarvis is in Focus
def is_jarvis_in_focus():
    active_window = gw.getActiveWindow()
    if active_window:
        return "Jarvis" in active_window.title
    return False

# Start Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam. Exiting...")
    exit()

while True:
    if not is_jarvis_in_focus():
        print("Jarvis is not in focus. Pausing detection...")
        cv2.waitKey(500)  # Wait before checking again
        continue

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        frame_count += 1

        if frame_count % 30 != 0:  # Analyze every 30th frame (performance boost)
            continue

        face_roi = frame[y:y+h, x:x+w]  # Extract only the face region

        # Draw Rectangle around Face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        try:
            # Analyze Mood
            analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            mood = analysis[0]['dominant_emotion']

            # Display Mood on Screen
            cv2.putText(frame, f'Mood: {mood}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Jarvis Responds Only if Mood Changes
            if mood != last_mood:
                print(f"Jarvis: You seem to be {mood} today.")

                if mood in mood_songs and mood not in song_played:
                    song_played.add(mood)  # Mark mood as played
                    jarvis_speak(f"Here is a song as per your mood, {mood}. Enjoy!")
                    webbrowser.open(random.choice(mood_songs[mood]))  # Play random song

                last_mood = mood

        except Exception as e:
            print(f"Detection Error: {e}")

    cv2.imshow('Jarvis Emotion Detector', frame)

    # Press 'q' to Exit (Jarvis will confirm shutdown)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        jarvis_speak("Goodbye! Shutting down.")
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
