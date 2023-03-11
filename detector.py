import cv2
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import numpy as np
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
import pandas as pd
import time
import os
import statistics
import webbrowser




model = load_model('Model/data/cnn_model.h5')

# Load CSV file containing links for different emotions
df = pd.read_csv('Spotify/data/spotify_tracks.csv')

# Create subsets for each label/emotion
subsets = {}
for label in df['label'].unique():
    subset = df.loc[df['label'] == label]
    subsets['sub_' + label] = subset

# Create dictionary mapping emotions to links
emotion_links_dict = dict(zip(df['label'], df['track_link']))


def detect_emotion():
    frame_window = st.image([])
    # define dictionary mapping labels to emotions
    emotion_dict = {
        0: "Angry",
        1: "Happy",
        2: "Sad",
        3: 'Neutral'
    }
    # initialize video capture from webcam
    cap = cv2.VideoCapture(0)

    #dict_ = {}
    new_list = []

    #start_time = time.time()

    while (cap.isOpened()):
        # read frame from video capture
        ret, frame = cap.read()
        if ret:
            # convert color frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # load pre-trained Haar Cascade classifier for face detection
            face_cascade = cv2.CascadeClassifier('Model/data/haarcascade_frontalface_default.xml')

            # detect faces in grayscale image using the Haar Cascade classifier
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # loop over detected faces
            for (x, y, w, h) in faces:
                # extract the face region from the grayscale image
                roi_gray = gray[y:y+h, x:x+w]

                # resize the face region to 48x48 pixels to match the input size of the model
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                roi_gray = roi_gray.reshape((48, 48, 1))

                # preprocess the face region by normalizing pixel values and reshaping
                roi_gray = np.expand_dims(roi_gray, axis=-1)
                roi_gray = np.expand_dims(roi_gray, axis=0)
                roi_gray = roi_gray.astype('float32') / 255.0

                # make emotion prediction using the pre-trained model
                predictions = model.predict(roi_gray)
                label = np.argmax(predictions)
                emotion = emotion_dict[label]
                new_list.append(emotion)


                # draw rectangle around detected face 
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Draw the predicted emotion label on the frame
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # display the video stream with detected faces and predicted emotions
            frame_window.image(frame)

            if len(new_list) > 18:
                non_neutral = [emotion for emotion in new_list if emotion != 'Neutral']
                if len(non_neutral) == 0:
                    emotion = 'neutral'
                else:
                    emotion = statistics.mode(non_neutral).lower()
                return emotion

            # break loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print('Could not read frame from video capture')
    return emotion


