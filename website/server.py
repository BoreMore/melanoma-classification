from flask import Flask, request
from flask_cors import CORS
import json
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB5
import cv2

model = tf.keras.models.load_model('../merged_model.h5')

scaler = preprocessing.MinMaxScaler()

app = Flask(__name__)
CORS(app)


# data preprocessing functions
def clean_data(df, cols_to_del):
    for col in cols_to_del:
        if col in df.columns:
            del df[col]
    
    # replace NA in this col with mode
    #mode = df['anatom_site_general_challenge'].mode()[0]
    #df['anatom_site_general_challenge'].fillna(mode, inplace=True)
    
    # drop NA age and sex
    df = df.dropna(axis=0, subset=['age_approx'])
    
    return df

def hot_encode(df, columns):
    for col in columns:
        if col in df.columns:
            one_hot = pd.get_dummies(df[col])
            df = df.drop(col, axis = 1)
            df = df.join(one_hot)
    
    return df

def scale(df, train=True):
    df_scaled = df.copy()
    if 'image_name' in df.columns:
        del df_scaled['image_name']
    
    if train:
        scaler.fit(df_scaled)
    df_scaled = scaler.transform(df_scaled)
    df_scaled = pd.DataFrame(df_scaled, columns=['age_approx', 'female', 'male'])
    if 'image_name' in df.columns:
        df_scaled['image_name'] = df['image_name'].tolist()
    
    return df_scaled


def prep_data():
    # call preprocessing functions on train and test datasets containing patient metadata
    df_train = pd.read_csv('../train.csv')
    # I dropped the anatom_site because values in this column were missing in the test data
    # in the future: try random forest or impute mode instead in both train and test data
    # target is contained in the tfrecords, so I dropped it here
    df_train = clean_data(df_train, ['patient_id', 'diagnosis', 'benign_malignant', 'anatom_site_general_challenge', 'target'])
    df_train = hot_encode(df_train, ['sex'])
    df_train = scale(df_train)


def shape_user_input(age, sex):
    male = 0
    female = 0
    # user input:
    if sex.lower() is 'male':
        male = 1.0
        female = 0.0
    elif sex.lower() is 'female':
        male = 0.0
        female = 1.0

    patient_data = {
        'age_approx': age,
        'female': female,
        'male': male
    }

    df_predict = pd.DataFrame(patient_data, index=[0], columns=['age_approx', 'female', 'male'])

    df_predict = scale(df_predict, train=False)
    return df_predict

import base64
def shape_image(image_name):
    #image = cv2.imread(image_name)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encoded_data = image_name.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    IMAGE_SIZE = [256, 256]
    image = tf.image.resize(image, [*IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0 
    image = tf.reshape(image, [-1, *IMAGE_SIZE, 3])
    return image


@app.route('/api/predict', methods=['POST'])
def run_prediction():
    if request.method == 'POST':

        df_predict = shape_user_input(request.form['age'], request.form['sex'])
        image = shape_image(str(request.form['image_name']))
        prediction = model.predict([df_predict, image])
        print(prediction)
        return str(prediction)


if __name__ == '__main__':
    prep_data()
    app.run(debug=True)