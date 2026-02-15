import cv2
from deepface import DeepFace
import pandas as pd
from tkinter import *
from tkinter import messagebox, ttk
import threading
import webbrowser
import matplotlib.pyplot as plt
import time

# ---------------- DATASET ----------------
def load_dataset():
    try:
        return pd.read_csv("music_dataset.csv")
    except Exception as e:
        messagebox.showerror("Dataset Error", str(e))
        return None


# ---------------- GLOBAL VARIABLES ----------------
music_df = load_dataset()

emotion_history = []
emotion_timeline = []

current_song = None
current_artist = None

last_emotion = "Detecting..."
last_confidence = 0
frame_counter = 0

camera_running = False


# ---------------- CAMERA INIT ----------------
def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(1)
    return cap


# ---------------- MUSIC RECOMMENDATION ----------------
def recommend_music(emotion):
    global current_song, current_artist

    songs = music_df[music_df["Emotion"] == emotion]

    if not songs.empty:
        row = songs.sample(1).iloc[0]
        current_song = row["Song"]
        current_artist = row["Artist"]
        return current_song, current_artist

    return "No Song Found", "No Artist Found"


def update_recommendation_from_emotion(emotion, confidence):
    song, artist = recommend_music(emotion)

    emotion_label.config(
        text=f"Emotion: {emotion} ({confidence}%)"
    )
    song_label.config(text=f"🎵 Song: {song}")
    artist_label.config(text=f"🎤 Artist: {artist}")


# ---------------- EMOTION DETECTION (SINGLE CAPTURE) ----------------
def detect_emotion():
    global camera_running

    if camera_running:
        return None, None

    cap = open_camera()

    if not cap.isOpened():
        messagebox.showerror("Error", "Camera not accessible")
        return None, None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, None

    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False
        )

        if isinstance(result, list):
            result = result[0]

        emotion = result["dominant_emotion"]
        confidence = result["emotion"][emotion]

        return emotion.capitalize(), round(confidence, 2)

    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None, None


# ---------------- REALTIME DETECTION ----------------
def realtime_emotion_detection():
    global frame_counter, last_emotion, last_confidence, camera_running

    if camera_running:
        return

    camera_running = True
    cap = open_camera()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        frame_counter += 1

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            # Run DeepFace every 20 frames
            if frame_counter % 20 == 0:
                try:
                    result = DeepFace.analyze(
                        face_img,
                        actions=["emotion"],
                        enforce_detection=False
                    )

                    if isinstance(result, list):
                        result = result[0]

                    emotion = result["dominant_emotion"]
                    confidence = result["emotion"][emotion]

                    last_emotion = emotion.capitalize()
                    last_confidence = round(confidence, 2)

                    emotion_history.append(last_emotion)
                    emotion_timeline.append(last_emotion)

                    # Update UI safely
                    root.after(
                        0,
                        update_recommendation_from_emotion,
                        last_emotion,
                        last_confidence
                    )

                except:
                    pass

            label = f"{last_emotion} ({last_confidence}%)"

            cv2.rectangle(frame, (x, y), (x+w, y+h),
                          (255, 0, 0), 2)

            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 0, 0), 2)

        cv2.imshow("Real-Time Emotion Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(
                "Real-Time Emotion Detection",
                cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    camera_running = False


# ---------------- OPEN MUSIC ----------------
def open_spotify():
    if current_song:
        webbrowser.open(
            f"https://www.google.com/search?q={current_song}+{current_artist}+spotify"
        )


def open_youtube():
    if current_song:
        webbrowser.open(
            f"https://www.google.com/search?q={current_song}+{current_artist}+youtube"
        )


# ---------------- GRAPH ----------------
def show_emotion_trend():
    if not emotion_timeline:
        messagebox.showinfo("Info", "No emotion data available")
        return

    emotion_map = {
        "Angry": 1, "Disgust": 2, "Fear": 3,
        "Sad": 4, "Neutral": 5,
        "Happy": 6, "Surprise": 7
    }

    y = [emotion_map.get(e, 0) for e in emotion_timeline]
    x = list(range(len(y)))

    plt.figure()
    plt.plot(x, y)
    plt.title("Emotion Trend Over Time")
    plt.xlabel("Detection Step")
    plt.ylabel("Emotion")
    plt.yticks(list(emotion_map.values()),
               list(emotion_map.keys()))
    plt.show()


# ---------------- AUTO DETECTION ----------------
def get_recommendation():
    if camera_running:
        return

    emotion, confidence = detect_emotion()

    if emotion:
        emotion_history.append(emotion)
        emotion_timeline.append(emotion)
        update_recommendation_from_emotion(emotion, confidence)

    root.after(10000, start_emotion_detection)


def start_emotion_detection():
    threading.Thread(
        target=get_recommendation,
        daemon=True
    ).start()


# ---------------- UI ----------------
root = Tk()
root.title("Emotion-Based Music Recommendation")
root.geometry("800x600")

title_label = ttk.Label(
    root,
    text="Emotion-Based Music Recommendation",
    font=("Helvetica", 22, "bold")
)
title_label.pack(pady=20)

emotion_label = ttk.Label(root, text="Emotion: None",
                          font=("Helvetica", 16))
emotion_label.pack(pady=5)

song_label = ttk.Label(root, text="🎵 Song: None",
                       font=("Helvetica", 16))
song_label.pack(pady=5)

artist_label = ttk.Label(root, text="🎤 Artist: None",
                         font=("Helvetica", 16))
artist_label.pack(pady=5)

button_frame = ttk.Frame(root)
button_frame.pack(pady=20)

ttk.Button(button_frame, text="Detect Emotion",
           command=start_emotion_detection).grid(row=0, column=0, padx=10)

ttk.Button(button_frame, text="Open Spotify",
           command=open_spotify).grid(row=0, column=1, padx=10)

ttk.Button(button_frame, text="Open YouTube",
           command=open_youtube).grid(row=0, column=2, padx=10)

ttk.Button(button_frame,
           text="Start Real-Time Detection",
           command=lambda: threading.Thread(
               target=realtime_emotion_detection,
               daemon=True).start()
           ).grid(row=1, column=0, pady=10)

ttk.Button(root,
           text="Show Emotion Trend Graph",
           command=show_emotion_trend).pack(pady=10)

root.mainloop()