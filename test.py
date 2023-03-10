import cv2
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import numpy as np
from tensorflow.keras.models import load_model


# Define Spotify credentials
client_id = '5237554f6e9545d5bfa9b4d6499acc37'
client_secret = 'e7b63f73c83745cf9ad9f49ebadb70d5'
redirect_uri = 'http://localhost:8501/callback'
scope = 'user-library-read user-read-playback-state user-modify-playback-state'

# Define emotion recognition function
def detect_emotion():

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
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # display the video stream with detected faces and predicted emotions
            cv2.imshow('Emotion Detection', frame)

        # break loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # release video capture and close all windows
        cap.release()
        cv2.destroyAllWindows()

        return emotion


# Define function to open Spotify song based on detected emotion
def open_spotify(emotion):
    # Create Spotify API client
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                                   client_secret=client_secret,
                                                   redirect_uri=redirect_uri,
                                                   scope=scope))

    # Define search query based on detected emotion
    if emotion == 'Happy':
        query = 'mood happy'
    elif emotion == 'Sad':
        query = 'mood sad'
    elif emotion == 'Angry':
        query = 'mood angry'
    elif emotion == 'Neutral':
        query = 'mood chill'
    else:
        query = 'mood relax'

    # Search for song on Spotify
    results = sp.search(q=query, type='track', limit=1)

    # Get song ID
    song_id = results['tracks']['items'][0]['id']

    # Play song on user's device
    sp.start_playback(uris=[f'spotify:track:{song_id}'])



# Define Streamlit app
def main():
    # Set page title and icon
    st.set_page_config(page_title='Moodify', page_icon=':musical_note:')

    # Load video for emotion recognition
    st.video(0)




# Define function to get Spotify track ID based on emotion
def get_track_id(emotion):
    if emotion == "Happy":
        # Get happy songs from Spotify
        results = sp.search(q='genre:happy', type='track', limit=1)
        # Get Spotify track ID from results
        track_id = results['tracks']['items'][0]['id']
    elif emotion == "Sad":
        # Get sad songs from Spotify
        results = sp.search(q='genre:sad', type='track', limit=1)
        # Get Spotify track ID from results
        track_id = results['tracks']['items'][0]['id']
    elif emotion == "Angry":
        # Get angry songs from Spotify
        results = sp.search(q='genre:angry', type='track', limit=1)
        # Get Spotify track ID from results
        track_id = results['tracks']['items'][0]['id']
    else:
        # Get neutral songs from Spotify
        results = sp.search(q='genre:lofi', type='track', limit=1)
        # Get Spotify track ID from results
        track_id = results['tracks']['items'][0]['id']
    return track_id

# Define function to play Spotify track based on emotion
def open_spotify(emotion):
    # Get Spotify track ID based on emotion
    track_id = get_track_id(emotion)
    # Play Spotify track
    sp.start_playback(uris=['spotify:track:' + track_id])





import cv2
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import numpy as np
from tensorflow.keras.models import load_model


# Set page title and icon
st.set_page_config(page_title='Moodify Me', page_icon=':musical_note:')


#create global var for video frame
frame_window = st.image([])


model = load_model('Model/data/cnn_model.h5')

def detect_emotion():
    global frame_window
    # define dictionary mapping labels to emotions
    emotion_dict = {
        0: "Angry",
        1: "Happy",
        2: "Sad",
        3: 'Neutral'
    }
    # initialize video capture from webcam
    cap = cv2.VideoCapture(0)

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

            # break loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print('Could not read frame from video capture')
    return emotion

# Define function to open Spotify song based on detected emotion
def open_spotify(emotion):
        
    # Define search query based on detected emotion
    if emotion == 'Happy':
        query = 'mood happy'
    elif emotion == 'Sad':
        query = 'mood sad'
    elif emotion == 'Angry':
        query = 'mood angry'
    elif emotion == 'Neutral':
        query = 'mood chill'
    else:
        query = 'mood relax'

    # Search for song on Spotify
    results = sp.search(q=query, type='track', limit=1)

    # Get song ID
    song_id = results['tracks']['items'][0]['id']

    # Play song on user's device
    sp.start_playback(uris=[f'spotify:track:{song_id}'])




def main():
    global frame_window
  
    # Detect emotion from video stream
    detect_emotion()

    # Add button to play Spotify track based on detected emotion
    if st.button('Moodify Me'):
        # Get detected emotion
        detected_emotion = st.session_state['emotion']
        # Play Spotify track based on detected emotion
        open_spotify(detected_emotion)
    


btn = st.button("Moodify Me")

if btn:
	if not(emotion):
		st.warning("Please let me capture your emotion first")
		st.session_state["run"] = "true"
	else:
		webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
		np.save("emotion.npy", np.array([""]))
		st.session_state["run"] = "false"


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    detect_emotion()


#st.header("Emotion Based Music Recommender")
#header_image = cv2.imread('imgs/streamlit-header_temp.png')
#st.image(header_image, use_column_width=True)
