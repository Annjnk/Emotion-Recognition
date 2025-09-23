Emotion recognition

INPUT: Using web camera to do the face detection,
Haar cascade classifier detects teh coordinates of the face in each frame

Detection converted to greyscale and resized to 64x64 pixels, pixel values are normalised to [0,1] by dividing by 255
Data reshaped to match the model input (batch_size, 64, 64, 1)

MODEL: Using Keras model trained on the fer2013 dataset
Input-64x64 grayscale face image
Output-probabilites of 8 emotions (neutral, happiness, surprise, etc)

Can analyses the images and looks for patterns in facial features like eyes, mouth, eyebrows to determine emotions.

OUTPUT: Each emotions probability is shown as a coloured bar with text above the face, probabilities update live, every frame (30fps)
Press q to exit

The new env: python3 -m venv emotion-env
source emotion-env/bin/activate
python real_time_video_barchart.py
