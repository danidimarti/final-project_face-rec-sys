import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import base64

st.markdown("<h1 style='color:#9FE1B4; font-family: serif;'>The CNN Model</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='color:#9FE1B4;'>Deep learning model for image classification</h3>", unsafe_allow_html=True)

st.write(
    "<div style='font-size:1.2rem'>"
    "A machine needs data to learn. Lots of it. Data is the most important part of any machine learning / deep learning project, because the model will be nothing more than a product of the data we used to trained it. This this task I have used a image dataset found on <a href='https://www.kaggle.com/datasets/msambare/fer2013'>Kaggle</a>.\n\
    <br><br>The *pixel* column of the df contain the pixel values of each image. There total 96 pixel values associated with each image because each image is grey-scaled and of resolution 48x48.</div>",
    unsafe_allow_html=True
)



# Display an image from a URL
st.image('imgs/fer2013_sample.png', caption='Fer2013_Dataframe', use_column_width=True)


st.write(
    "<div style='font-size:1.2rem;'>"
    "<br>Now, let\'s check the number of emotion categories we have the number of images associated with each:</div>",
    unsafe_allow_html=True
)

# Display an image from a URL
st.image('imgs/fer2013_exploration_bars.png', caption='Fer2013_Emotion Count', use_column_width=True)

st.markdown(">There are 7 categories of emotions in this data-set and emotion disgust has the minimum images around 5&ndash;10% of other classes."
)

st.write(
    "<div style='font-size:1.2rem'>"
    "<br>Let\'s visualize the images of each emotion category:</div>",
    unsafe_allow_html=True
)

# Display an image from a URL
st.image('imgs/Fer2013_imgs_example.png', caption='Fer2013_Images', use_column_width=True)

st.write(
    "<div style='font-size:1.7rem; color:#FFFFE0;'>"
    "Fer2013 summary analysis:</div>",
    unsafe_allow_html=True
)

# Define the HTML string with the ordered list tags
html_string = "<div style='font-size:1.2rem;'><ol type='1'><li>The data contains a wide range of images like, male, female, kids, olds, white, black etc.</li><li>It contains some non-human images, like cartoons(first row, last column)</li><li>The dataset contain images collected with different lighting and angles.</li></ol></div>"

# Display the HTML string in the Streamlit app
st.write(html_string, unsafe_allow_html=True)

st.markdown("> I have decided to train my model using the most 'distinguishable' emotions on the dataset 0:Anger, 3:Happy, 4:Sad and 6:Neutral. They are also the emotions with the higher number of images.")


st.markdown("<h2 style='color:#FFA500; font-family: serif;'>Creating the Model</h2>", unsafe_allow_html=True)


st.write(
    "<div style='font-size:1.7rem; color:#FFFFE0;'>"
    "Steps:</div>",
    unsafe_allow_html=True
)
# Define the HTML string with the ordered list tags
html_string = "<div style='font-size:1.2rem;'><ol type='1'><li>Ensure the data is compatible with the model needs: h:48 x w:48 x color:1 (greyscale) .</li><li>Stack the images so we can use mini-batch gradient descent as optimizer (system of small batches and feedback loops. It's less computationally efficient than SGD but more stable.)</li><li>Label encode categories so they are compatible with the model.</li><li>Split the data into training and validation/test set.</li><li> Normalize the image arrays, because neural networks are highly sensitive to non-normalize data.</ol></div>"

# Display the HTML string in the Streamlit app
st.write(html_string, unsafe_allow_html=True)

st.write(
    "<div style='font-size:1.7rem; color:#FFFFE0;'>"
    "Model Settings:</div>",
    unsafe_allow_html=True
)


# Define the HTML string with the nested ordered list
html_string = "<div style='font-size:1.2rem;'><ol><li>Shuffling and Stratification: split the data into random order and make sure that all classes are being represented in the split.</li><li>Model Operation Layers:<ol><li>Conv2D: applies filter to extract features that are spacially related</li><li>BatchNormalization: normalizes the inputs of the previous layer to speed training and improve performance</li><li>Dense: connected the neurons of the previous layers with the ones of the current layer</li><li>Dropout: randomly drops out some neurons during training to prevent overfitting</li></ol></li><li>Activation Function: ELU. applied to the output. allows for negative values to pass through the neural networks without being ignored (better performance). Avoids Relu problems where neurons can become dead and decrease in accuracy.</li><li>Callbacks: list of functions that will be called during the training to improve performance.<ol><li>EarlyStopping: avoids over-fitting</li><li>ReduceLROnPlateau: reduce learning rate when the validation accuracy plateaus.</li><li>ImageDataGenerator: applies changes to the image (e.g. rotations)</li></ol><li>Batch Size: 32</li><li>Epochs: 30</li><li>Optimizer: Adam. Commonly used for training images and speech rec.</li></li></ol></div>"

# Display the HTML string in the Streamlit app
st.write(html_string, unsafe_allow_html=True)

file_ = open("imgs/training-model.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="model-gif" width="695" height="385">',
    unsafe_allow_html=True
)


st.write(
    "<div style='font-size:1.2rem'>"
    "<br>Plotting training and validation metrics</div>",
    unsafe_allow_html=True
)

# Display an image from a URL
st.image('imgs/acc_loss_overtime.png', caption='Accuracy and Loss', use_column_width=True)

st.markdown(">The epoch\'s history shows that the accuracy gradually increases, reaching +73% on training and +75% on validation data. We also see a gradual decrease in less, with a sudden spike around epoch 5. This could be a signed of overfitting or unstable learning, but we can see that the validation data goes back to normal later, likely regularized by the Dropout layer or the ReduceLROnPlateau optmizer.")


# Display an image from a URL
st.image('imgs/acc_loss_value.png', caption='Accuracy and Loss', use_column_width=True)

st.markdown(">analysis here.")

st.image('imgs/validation_metrics.png', caption='Accuracy and Loss', use_column_width=True)

st.markdown(">analysis here.")

st.image('imgs/confusion_matrix.png', caption='Accuracy and Loss', use_column_width=True)

st.markdown(">analysis here.")

st.image('imgs/true_vs_pred_images.png', caption='Accuracy and Loss', use_column_width=True)

st.markdown(">CONCLUSION: IT IS VERY HARD FOR BOTH HUMANS AND COMPUTERS TO READ A RESTING BITCH FACE.")