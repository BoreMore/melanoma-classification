# melanoma-classification
Neural network trained on Kaggle dataset to predict malignant melanomas

Created 6/6/2020

## Overview
The challenge was to create a model that could predict whether or not a patient had melanoma based on patient metadata such as age, sex, and the anatomical site of a mole and/or an image of a mole. This approach used both patient metadata - the age and sex of a patient - and an image of a mole.

## Data Preprocessing
Out of the various data the tabular data (patient metadata) provided, only the patient age and sex was used. All other columns were dropped from the training data. The anatomical site data would likely have improved the accuracy of the model, but the exploratory data analysis showed that some anatomical site data was missing in the test data. Using random forest or the mode to impute missing data may be a viable solution to this.

The sex column was one-hot encoded for both the train and test data since it contained categorical values. The age column was scaled down to values between 0 and 1 for both the train and test data using a sklearn preprocessing MinMaxScaler.

Additionally, images were read from tfrecords files and converted to uint8 tensors containing the RGB image channels. Therefore, each image was represented as a multidimensional array (technically, tensors aren't multidimensional arrays, but they can be thought of as such) containing values between 0 and 255. These values were each then divided by 255 to normalize the image data and produce a float between the values of 0.0 and 0.1. The image data was also resized and reshaped to a consistent size and shape that could be passed in to the neural network as input.

Afterwards, the image data was augmented to produce variation in the training data. Only a random flip to the left or right was used. More complex image augmentation techniques may improve the model's accuracy.

## Neural Network
Tensorflow (including the tf.keras module) was used to create a merged neural network. The Model class and Functional API of tf.keras were mainly used to create the neural network. Two neural networks were created and their outputs were merged using the Keras Concatenate layer. The first neural network that took the patient metadata in as an input consisted only of Dense layers. The second neural network was a CNN that utilized transfer learning. It used EfficientNetB5 with the weights of ImageNet. The model was primarily created and ran on Kaggle notebooks.

## Evaluation
This competition's submissions were evaluated based on AUC (area under the ROC curve). The ROC (receiver operating characteristic) curve is based on the true positive rate and false positive rate of predictions. 

The highest AUC value possible is 1, and the lowest AUC value possible is 0. A value of 1 suggests that the model's predictions are right all the time, and a value of 0 suggests that the model's predictions are always wrong. A value of 0.5 suggests that the model's predictions are random and that it has no classification abilities. 

The competition submission associated with this model received a private score of 0.7131 (scored on 70% of the test data) and a public score of 0.7045 (scored on the other 30% of the test data).

## Website
The model is deployed on a website using jQuery and Bootstrap on the frontend and Flask on the backend to serve as an API for the machine learning model. To start the server, enter the website directory and run ```python server.py```. However, you will need to acquire the model by running the other notebooks provided because it is too large to upload to GitHub. THIS MODEL SHOULD NOT BE USED FOR MEDICAL DIAGNOSES.

## Dataset
The dataset used for this neural network was part of the SIIM-ISIC Melanoma Classification prediction competition on Kaggle. The competition and data can be found here:
https://www.kaggle.com/c/siim-isic-melanoma-classification/data
