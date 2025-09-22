import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = 'fer_model.h5'  # Make sure this file exists
EMOTIONS = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']

# Improved bar settings for visibility
BAR_WIDTH = 250
BAR_HEIGHT = 25
BAR_MARGIN = 10
BAR_COLOR = (0, 255, 0)        # Bright green
BG_COLOR = (0, 0, 0)           # Black background for bars
TEXT_COLOR = (255, 255, 255)   # White text
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2

# ---------------------------
# LOAD MODEL
# ---------------------------
model = load_model(MODEL_PATH, compile=False)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------------------------
# START WEBCAM
# ---------------------------
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow("Emotion Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Emotion Recognition", 800, 600)

# ---------------------------
# REAL-TIME LOOP
# ---------------------------
while True:
    ret, frame = camera.read()
    if not ret:
        print("Cannot read camera frame. Check camera permissions.")
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (64, 64))      # Model expects 64x64
        roi = roi.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=-1)   # Add channel dimension
        roi = np.expand_dims(roi, axis=0)    # Add batch dimension

        preds = model.predict(roi, verbose=0)[0]

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Draw emotion bars above the face
        for i, (emotion, prob) in enumerate(zip(EMOTIONS, preds)):
            bar_x = x
            bar_y = y - (len(EMOTIONS)-i)*(BAR_HEIGHT+BAR_MARGIN) - 10

            # Draw background rectangle
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x+BAR_WIDTH, bar_y+BAR_HEIGHT), BG_COLOR, -1)

            # Draw probability bar
            bar_length = int(prob * BAR_WIDTH)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_length, bar_y+BAR_HEIGHT), BAR_COLOR, -1)

            # Draw text
            cv2.putText(frame, f"{emotion}: {prob*100:.1f}%", 
                        (bar_x + 5, bar_y + BAR_HEIGHT - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

    cv2.imshow("Emotion Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

