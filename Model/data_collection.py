
# mediapipe > library for detecting holistic movements and hand gestures
import mediapipe as mp
#opencv (cv2) > library for capturing and processing images in real-time
import cv2
import numpy as np


#Creating a capture object to get the frames from the camera. 
# arg=0 > specifies we want to use the default camera from the computer. 
cap = cv2.VideoCapture(0)

name = input('Enter the name of the data: ')

#Creating objects for detecting holistic movements and hand gestures
holistic = mp.solutions.holistic
hands = mp.solutions.hands

#Initializing the holistic detection object. This object is used later in the loop to process the frames.
holis = holistic.Holistic()

#Creating a drawing_utils object which is used to draw the landmarks and connections on the output image.
drawing = mp.solutions.drawing_utils

X = []
data_size = 0


while True:

    lst = []

    #reading the frame
    #The underscore is used to ignore the return value of the method which is a Boolean value indicating whether the frame was read successfully or not
    _, frm = cap.read()

    #flipping the frame horizontally to get a mirror image of the current frame
    frm = cv2.flip(frm,1)

    #process frame to detect holistic movements of a person in the frame. 
    #conver BGR to RGB for better image processing and better interpretation of other libraries (e..g Mediapipe and Matplotlib)
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # check if the landmarks for the face and hands are in the frame. 
        #if detected: append normalize X and Y coords of each landmark to list
            #normalize landmark: by subtrating the coords of the first landmark, which is the nose for the face and index finger for the hands
        #if hands NOT detected: append 42 zeros for each hand, which correspondos to 21 landmarks
    
    if res.face_landmarks: 
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)        
        else:
            for i in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        # append the lst to X which contains the landmarks for all frames. 
        X.append(lst)
        #data_size keeps track of the number of frames processed.
        data_size = data_size+1

    #using drawing_utils object to draw the landmarks and connections to the frame.
    #the resulting frame is displayed with the cv2.imshow()
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    #defining the size of the window
    cv2.namedWindow('window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('window', 640, 480)
    
    #tell the user how much data we have collected
    cv2.putText(frm, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    # display camera frame named 'WINDOW'
    cv2.imshow('window', frm)

    # Destroy windows and releassing after (1) millisecond if user presses ESC (in ASCII ESC = 27  )
    if cv2.waitKey(1) == 27 or data_size > 99:
        cv2.destroyAllWindows()  # Corrected function name
        cap.release()
        break

# .npy saves the data in array format. used for manipulating largem, multidimensional data arrays and matrices
np.save(f'img_collection/{name}.npy', np.array(X))
print(np.array(X).shape)

