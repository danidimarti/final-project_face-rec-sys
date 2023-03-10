import cv2
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import numpy as np
from tensorflow.keras.models import load_model
import os
import time
import pandas as pd


st.image('imgs/streamlit-header_temp.png', use_column_width=True)

#create global var for video frame
#frame_window = st.image([])

model = load_model('Model/data/cnn_model.h5')

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

    dict_ = {}
    start_time = time.time()

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

                # draw rectangle around detected face 
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Draw the predicted emotion label on the frame
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # display the video stream with detected faces and predicted emotions
            frame_window.image(frame)

            # break loop if 'q' key is pressed or time limit of 2 seconds is exceeded
            if (cv2.waitKey(1) & 0xFF == ord('q')) or (time.time() - start_time >= 2):
                break
        else:
            print('Could not read frame from video capture')

    cap.release()
    cv2.destroyAllWindows()
    return emotion


if st.button('Moodify me'):
    emotion = detect_emotion()
    st.write(f"It is a great day to feel {emotion}!")
    
    

def open_link(emotion):
    df = pd.read_csv('links.csv')
    link = df[df['Emotion'] == emotion]['Link'].values[0]
    st.write(f"Opening link for {emotion}: {link}")
    # Open link here



#1 set timer of two seconds to capture emotion
#2.once emotion is capture, make the function stop (stop the loo)
#3. save the result (detected emotion) to be used later.  


# calculate 

# 1. Make the function stop
    # 1. After emotion is recognized
    # 2. Calculate time until next emotion
    # 3. If time is greater than X
    # 4. Stop the loop: functions stops
    # 5. You have a result

# 2. Save it somwehere else and import it here
# 3. debug the import
# 4. Add the next step

# Set page title and icon
#st.set_page_config(page_title='Moodify Me', page_icon=':musical_note:')


#st.header("Emotion Based Music Recommender")


