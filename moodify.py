import cv2
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp 
import webbrowser


# load pre-trained emotion detection model
model = load_model('Model/data/cnn_model.h5')

# define dictionary mapping labels to emotions
emotion_dict = {
    0: "Angry",
    1: "Happy",
    2: "Sad",
    3: 'Neutral'
}
# initialize video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    # read frame from video capture
    ret, frame = cap.read()

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
        print(roi_gray)
        
        roi_gray = roi_gray.reshape((48, 48, 1))
        #########

        # preprocess the face region by normalizing pixel values and reshaping
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = roi_gray.astype('float32') / 255.0


        # make emotion prediction using the pre-trained model
        predictions = model.predict(roi_gray)
        label = np.argmax(predictions)
        emotion = emotion_dict[label]

        # draw rectangle around detected face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # display the video stream with detected faces and predicted emotions
    cv2.imshow('Emotion Detection', frame)

    # break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release video capture and close all windows
cap.release()
cv2.destroyAllWindows()


####---- Streamlit App ---- ####

# Set page width to maximum
st.set_page_config(page_title='Moodify', page_icon=':smiley:', layout='wide')

# Insert header image
st.image('imgs\streamlit-header_temp.png', use_column_width=False)

# Center all elements of the page
st.markdown(
    f"""
    <style>
    .stApp {{
        max-width: 100%;
        margin: auto;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)