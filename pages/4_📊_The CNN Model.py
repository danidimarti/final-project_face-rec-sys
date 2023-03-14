import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import base64


# Define the CSS style for the container
container_style = """
    .text-container {
        background-color: #212121;
        border: 1px solid #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        color: #D3D3D3;
        font-size: 1.1rem;
        font-family: 'Open Sans', sans-serif;
        
    }
"""

bullet_style = """
    .bullet {
        background-color: #212121;
        border-radius: 10px;
        padding: 20px;
        color: #D3D3D3;
        font-size: 1.3rem;
        font-family: 'Open Sans', sans-serif;
        
    }
"""


with open('style.css') as f: 
    st.markdown(f'<style>{f.read()} </style>', unsafe_allow_html=True)

st.markdown("<h1 style='color:#9FE1B4; font-family: serif;'>The CNN Model</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='color:#9FE1B4;'>Deep learning model for image classification</h3>", unsafe_allow_html=True)

st.write(
    "<div style='font-size:1.2rem'>"
    "A machine needs data to learn. Lots of it. Data is the most important part of any machine learning / deep learning project, because the model will be nothing more than a product of the data we used to trained it. This this task I have used a image dataset found on <a href='https://www.kaggle.com/datasets/msambare/fer2013'>Kaggle</a>.\n\
    <br><br>The <i> pixel </i> column of the df contain the pixel values of each image. There total 96 pixel values associated with each image because each image is grey-scaled and of resolution 48x48.</div>",
    unsafe_allow_html=True
)



# Display an image from a URL
st.image('imgs/fer2013_sample.png', caption='Fer2013_Dataframe', use_column_width=True)


st.write(
    "<div style='font-size:1.2rem;'>"
    "<br>Now, let\'s check the number of emotion categories we have the number of images associated with each:</div>",
    unsafe_allow_html=True
)

st.image('imgs/fer2013_exploration_bars.png', caption='Fer2013_Emotion Count', use_column_width=True)

fer2013data = "There are 7 categories of emotions in this data-set and emotion disgust has the minimum images around 5&ndash;10% of other classes."

st.markdown(f'<div class="text-container"><style>{container_style}</style><div style="color:#FFDAB9; font-family: serif; font-size:1.5rem">Fer2013 data summary:</div>{fer2013data}</div><br><br>', unsafe_allow_html=True)


st.write(
    "<div style='font-size:1.2rem'>"
    "<br>Let\'s visualize the images of each emotion category:</div>",
    unsafe_allow_html=True
)

st.image('imgs/Fer2013_imgs_example.png', caption='Fer2013_Images', use_column_width=True)

###### ------ FER2013 SUMMARY ----- #####
html_string = "<div style='font-size:1.2rem;'><ol type='1'><li>The data contains a wide range of images like, male, female, kids, olds, white, black etc.</li><li>It contains some non-human images, like cartoons(first row, last column)</li><li>The dataset contain images collected with different lighting and angles.</li></ol></div>"


fer2013decision = f"{html_string}<br>I have decided to train my model using the most 'distinguishable' emotions on the dataset 0:Anger, 3:Happy, 4:Sad and 6:Neutral. They are also the emotions with the higher number of images."

st.markdown(f'<div class="text-container"><style>{container_style}</style><div style="color:#FFDAB9; font-family: serif; font-size:1.5rem">Fer2013 summary analysis:</div>{fer2013decision}</div><br><br>', unsafe_allow_html=True)


######### ------ CREATING THE MODEL - WHY THIS MODEL ------------- ########
st.markdown("<h2 style='color:#90EE90; font-family: serif;'>Creating the Model</h2>", unsafe_allow_html=True)

whythismodel = "<div style='font-size:1.2rem;'><div></div> I choose CNN because, although it might require a bit more fine tuning than other models it is a more general purpose model that is less computationally expensive and it doesn't requires extensive amount data to the trained (e.g. OpenFace, FaceNet).</div>"


st.markdown(f'<div class="bullet-points"><style>{bullet_style}</style><div style="color:#FFDAB9; font-family: serif; font-size:1.7rem">Why the CNN model?</div>{whythismodel}</div><br>', unsafe_allow_html=True)

######### ------- STEPS ------------- ########
steps = "<div style='font-size:1.2rem;'><ol type='1'><li>Ensure the data is compatible with the model needs: h:48 x w:48 x color:1 (greyscale) .</li><li>Label encode categories so they are compatible with the model.</li><li> Normalize the image arrays, because neural networks are highly sensitive to non-normalize data.</li><li>Stack the images so we can use mini-batch gradient descent as optimizer (system of small batches and feedback loops. It's less computationally efficient than SGD but more stable.)</li><li>Split the data into training and validation/test set.</li></ol></div>"


st.markdown(f'<div class="bullet-points"><style>{bullet_style}</style><div style="color:#FFDAB9; font-family: serif; font-size:1.7rem">Steps:</div>{steps}</div><br>', unsafe_allow_html=True)


######----- MODEL SETTINGS ------ ######

html_string = "<div style='font-size:1.2rem;'><ol><li>Shuffling and Stratification: split the data into random order and make sure that all classes are being represented in the split.</li><li>Model Operation Layers:<ol><li>Model ran on 23 layers</li><li>Conv2D: applies performance filter (250) to extract features that are spacially related</li><li>BatchNormalization: normalizes the inputs of the previous layer to speed training and improve performance</li><li>Dense: connected the neurons of the previous layers with the ones of the current layer</li><li>Dropout: randomly drops out some neurons during training to prevent overfitting</li></ol></li><li>Activation Function: ELU. applied to the output. allows for negative values to pass through the neural networks without being ignored (better performance). Avoids Relu problems where neurons can become dead and decrease in accuracy.</li><li>Callbacks: list of functions that will be called during the training to improve performance.<ol><li>EarlyStopping: avoids over-fitting</li><li>ReduceLROnPlateau: reduce learning rate when the validation accuracy plateaus.</li><li>ImageDataGenerator: applies changes to the image (e.g. rotations)</li></ol><li>Batch Size: 32</li><li>Epochs: 30</li><li>Optimizer: Adam. Commonly used for training images and speech rec.</li></li></ol></div>"

st.markdown(f'<div class="bullet-points"><style>{bullet_style}</style><div style="color:#FFDAB9; font-family: serif; font-size:1.7rem">Model Settings:</div>{html_string}</div><br><br>', unsafe_allow_html=True)


#### ----- GIF ----- ######
file_ = open("imgs/training-model.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="model-gif" width="695" height="385">',
    unsafe_allow_html=True
)

##### ---- PLOTTING ---- ##### 
st.write(
    "<div style='font-size:1.2rem'>"
    "<br>Plotting training and validation metrics</div><br><br>",
    unsafe_allow_html=True
)


###### ----- ACC AND LOSS OVER TIME ----- ######
st.write(
    "<div style='font-size:1.7rem; color:#FFFFFF;'>"
    "Accuracy and Loss over time:</div>",
    unsafe_allow_html=True
)

st.image('imgs/acc_loss_overtime.png', caption='Accuracy and Loss', use_column_width=True)

epoch_overtime = 'The epoch\'s history shows that the accuracy gradually increases, reaching +73% on training and +75% on validation data. We also see a gradual decrease in loss, with a sudden spike around epoch 5. This could be a signed of overfitting or unstable learning, but we can see that the validation data goes back to normal later, likely regularized by the Dropout layer or the ReduceLROnPlateau optmizer'

st.markdown(f'<div class="text-container"><style>{container_style}</style><div style="color:#FFDAB9; font-family: serif; font-size:1.5rem">Accuracy and loss over time summary</div>{epoch_overtime}</div><br><br>', unsafe_allow_html=True)



######----- ACC AND LOSS VIOLIN GRAPH ------ ######
st.write(
    "<div style='font-size:1.7rem; color:#FFFFFF;'>"
    "Accuracy and Loss distribution:</div>",
    unsafe_allow_html=True
)
# Display an image from a URL
st.image('imgs/acc_loss_value.png', caption='Accuracy and Loss', use_column_width=True)

violin_graph = 'The violin graph provides an extra layer of information about our dataset, telling us about the distribution (spread) of the accuracy and loss rates across the data. <br><br> In our dataset, we observed that 50% of the data falls between +60% and 70% accuracy for the training set, and 65%-75% accuracy for the validation and test sets. The loss rate is a bit high, at around 0.6. We need to determine if these values are enough for the problem we are trying to solve by looking at the performance metrics and other models benchmarks.'

# Add the CSS style to the page using st.markdown()
st.markdown(f'<div class="text-container"><style>{container_style}</style><div style="color:#FFDAB9; font-family: serif; font-size:1.5rem">Accuracy and loss distribution summary</div>{violin_graph}</div><br><br>', unsafe_allow_html=True)


######----- PERFORMANCE METRICS ------ ######
st.write(
    "<div style='font-size:1.7rem; color:#FFFFFF;'>"
    "Perfomance Metrics:</div>",
    unsafe_allow_html=True
)
caption_metrics = 'Precision measures the % of correctly predicted a label out of all predicted labels. (e.g. 0.8 of the labels were correctly id) /// Recall measures % of a correctly predicted label out of all the instances of that label (e.g. predicted 70% of the happy labels as happy) /// F1-score is indicates the balance between precision and recall. /// Support # instances of each class in the dataset. /// Weighted Avg takes into account the number of samples in each class. /// Macro Avg the average of all classes.'
st.image('imgs/validation_metrics.png', caption=caption_metrics, use_column_width=True)

analysis_metrics = 'The precision, recall and f1-scores suggests that the model\'s performance is generally good across all classes. However, it also highlights the differences in performance within each class. For example, class 1 (happiness), has the highest precision and recall scores, indicating that the model is able to accurately predict this class more often than any other. At the same time, class 2 (sadness) has a the lowest precision score, indicating that the model makes more false positive predictions for this class.<br><br> Although at first glance this number might indicate a slight imbalance in the dataset, I believe that the lower precision score in this is class results from the facial expression similarities between Sad and Neutral as we have already seen. In addition, the weighted avg. and the macro avg. are similar indicating that the dataset is fairly balanced. '

st.markdown(f'<div class="text-container"><style>{container_style}</style><div style="color:#FFDAB9; font-family: serif; font-size:1.5rem">Performance metrics summary</div>{analysis_metrics}</div><br><br>', unsafe_allow_html=True)

######----- CONFUSION MATRIX & IMAGE COMPARISION ------ ######


st.image('imgs/confusion_matrix.png', caption='Confusion Matrix', use_column_width=True)

confusion_matrix = 'The confusion matrix table provides a more visual representation of the Model\'s performance by comparing the predictions with the true values of the dataset. It helps us identify where the model might need some improvements. There\'s definitely an opportunity to optimize the model by further balancing the data. That way the classes performance should be more evenly distributed. However, as we have seen, some images from the categories neutral and sad are hard to tell apart even for humans! <br><br> Finally, it is worth noting is that the model\'s predictions makes more mistakes differentiating the "Neutral" feeling to all other emotions. '

st.markdown(f'<div class="text-container"><style>{container_style}</style><div style="color:#FFDAB9; font-family: serif; font-size:1.5rem">Confusion matrix summary</div>{confusion_matrix}</div><br><br>', unsafe_allow_html=True)

st.image('imgs/true_vs_pred_images.png', caption='True Values vs Pred Values', use_column_width=True)

conclusion = 'The performance of this initial model will be sufficient to help us make inferences about the user\'s emotions via the up, but further optimizations are needed to fine-tuned it to a more optimal level. <br> <br> Overall, from this analysis we can conclude that it is hard for both humans and computers to read a resting b**** face. :)'
st.markdown(f'<div class="text-container"><style>{container_style}</style><div style="color:#FFDAB9; font-family: serif; font-size:1.5rem">Final Toughts</div>{conclusion}</div><br><br>', unsafe_allow_html=True)
