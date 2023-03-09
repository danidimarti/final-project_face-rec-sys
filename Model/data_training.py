import os
import numpy as np
import cv2
from keras.utils import to_categorical

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam


is_init = False
path = 'img_collection'
size = -1

label = []
dictionary = {}
#flag variable
c = 0

for i in os.listdir(path):
    file_path = os.path.join(path, i)  # Construct the full file path.
    if os.path.isfile(file_path) and i.split('.')[-1] == 'npy' and not(i.split('.')[0] == 'labels'):
        if not is_init:
            is_init = True
            X = np.load(file_path) 
            size = X.shape[0]
            y = np.array([i.split('.')[0]]*size).reshape(-1,1)
        else:
            X = np.concatenate((X, np.load(file_path)))
            y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1))) # removed .reshape(-1,1)
            
            
        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c = c+1

print(dictionary)
print(label)

print(y)
for i in range(y.shape[0]):
    y[i,0] = dictionary[y[i,0]]
y = np.array(y, dtype='int32')
print(y)

### hello = 0 nope = 1 ----> [1,0] ... [0,1]



# Slice the X array to have the same number of samples as y.
y = y[:X.shape[0], :]

# Convert y to one-hot encoded vectors.
y_onehot = to_categorical(y)

X_new = X.copy()
y_new = y.copy()
counter = 0

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt:
    X_new[counter] = X[i]
    y_new[counter] = y[i]
    counter = counter + 1

print(y)
print(y_new)

ip = Input(shape=(X.shape[1]))

m = Dense(512, activation='relu')(ip)
m = Dense(256, activation='relu')(ip)

op = Dense(y.shape[1], activation='softmax')(m)

# Define the neural network model.
model = Model(inputs=ip, outputs=op)

# Compile the neural network model and define accuracy metrics for review
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train the neural network model.
model.fit(X, y, epochs=100)

# Save the trained model and label list to disk.
model.save('data/model.h5')
np.save("data/labels.npy", np.array(label))