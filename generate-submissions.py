#!/usr/bin/env python
# coding: utf-8

# In[]:


# import necessary packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#!pip install fsspec --upgrade


from sklearn import preprocessing

#from keras.models import Sequential
#from keras.layers import Dense

import tensorflow as tf
import re


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Flatten, Concatenate, BatchNormalization
from tensorflow.keras.models import load_model
#!pip install efficientnet
from efficientnet.tfkeras import EfficientNetB5

from kaggle_datasets import KaggleDatasets
GCS_DS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')


# In[]:


# data preprocessing functions
scaler = preprocessing.MinMaxScaler()

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
    del df_scaled['image_name']
    
    if train:
        scaler.fit(df_scaled)
    df_scaled = scaler.transform(df_scaled)
    df_scaled = pd.DataFrame(df_scaled, columns=['age_approx', 'female', 'male'])
    df_scaled['image_name'] = df['image_name'].tolist()
    
    return df_scaled


# In[]:


# call preprocessing functions on train and test datasets containing patient metadata
df_train = pd.read_csv(GCS_DS_PATH + '/train.csv')
# I dropped the anatom_site because values in this column were missing in the test data
# in the future: try random forest or impute mode instead in both train and test data
df_train = clean_data(df_train, ['patient_id', 'diagnosis', 'benign_malignant', 'anatom_site_general_challenge', 'target'])
df_train = hot_encode(df_train, ['sex'])

df_train = scale(df_train)

df_test = pd.read_csv(GCS_DS_PATH + '/test.csv')
df_test = clean_data(df_test, ['patient_id', 'anatom_site_general_challenge'])
df_test = hot_encode(df_test, ['sex'])

df_test = scale(df_test, train=False)

print(df_train)
print(df_test)


# In[]:


# functions for reading from tfrecords
# code adapted from https://www.kaggle.com/ajaykumar7778/melanoma-tpu-efficientnet-b5-dense-head/output?select=submission_b5.csv

# get training and test datasets
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords/train*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords/test*.tfrec')

AUTO = tf.data.experimental.AUTOTUNE
IMAGE_SIZE = [256, 256]
BATCH_SIZE = 8

"""VALIDATION_SPLIT = 0.18
split = int(len(TRAINING_FILENAMES) * VALIDATION_SPLIT)

training_filenames = random.sample(TRAINING_FILENAMES, len(TRAINING_FILENAMES) - split)
VALIDATION_FILENAMES = random.sample(TRAINING_FILENAMES, split)
training_filenames, VALIDATION_FILENAMES = train_test_split(TRAINING_FILENAMES, test_size=split)
print(VALIDATION_FILENAMES)

print("Pattern matches {} data files. Splitting dataset into {} training files and {} validation files"
      .format(len(TRAINING_FILENAMES), len(training_filenames), len(VALIDATION_FILENAMES)))
TRAINING_FILENAMES = training_filenames"""

# scales and reshapes images
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, [*IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image    

# randomly flip image to introduce variation in dataset
# in the future: use more complex image modifications to introduce more variation in dataset
def augment_img(data):
    data['image'] = tf.image.random_flip_left_right(data['image'])
    return data

# counts how many images there are to set the steps per epoch
def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)    

# function I made to get associated metadata for each image
def mod_data(parsed_record, df):
    new_array = []
    
    image_names = parsed_record['image_name'].numpy().astype(str)
    idx = 0
    for x in image_names:
        # finds metadata that corresponds with the image name
        metadata = df.loc[df['image_name'].str.contains(x)]
        # removes image data if the tabular data row was dropped during preprocessing
        if metadata.empty:
            parsed_record['image'] = tf.constant(np.delete(parsed_record['image'].numpy(), idx, axis=0), dtype=tf.float32)
            parsed_record['image_name'] = tf.constant(np.delete(parsed_record['image_name'].numpy(), idx), dtype=tf.string)
            parsed_record['target'] = tf.constant(np.delete(parsed_record['target'].numpy(), idx), dtype=tf.int32)
        
        # adds metadata to array (to later be converted to tensor)
        else:
            del metadata['image_name']
            new_array.append(metadata.to_numpy())
            idx += 1

    # convert array containing metadata to tensor
    new_array = np.asarray(new_array).squeeze()
    tensor_array = tf.constant(new_array, dtype=tf.float64)
    parsed_record['metadata'] = tensor_array

    return parsed_record

# create training dataset
def parse_training_tfrecord(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image_name': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'target': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }
    
    dataset = tf.io.parse_single_example(example_proto, feature_description)
    dataset['image'] = decode_image(dataset['image'])
    dataset['target'] = tf.cast(dataset['target'], tf.int32)
    return dataset

# create testing dataset without 'target' feature
def parse_testing_tfrecord(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image_name': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }
    
    dataset = tf.io.parse_single_example(example_proto, feature_description)
    dataset['image'] = decode_image(dataset['image'])
    return dataset

# load dataset and map to features
def load_dataset(file_paths, labeled=True, ordered=False): 
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
        
    raw_dataset = tf.data.TFRecordDataset(file_paths, num_parallel_reads=AUTO)
    raw_dataset = raw_dataset.with_options(ignore_order)
    raw_dataset = raw_dataset.map(parse_training_tfrecord if labeled else parse_testing_tfrecord,
                                  num_parallel_calls=AUTO)
    return raw_dataset

# get the final training dataset to be used for the model
def get_training_dataset(file_paths, ordered=False):
    parsed_dataset_train = load_dataset(file_paths)
    
    parsed_dataset_train = parsed_dataset_train.map(augment_img, num_parallel_calls=AUTO)
       
    parsed_dataset_train = parsed_dataset_train.repeat()
    parsed_dataset_train = parsed_dataset_train.shuffle(2048)
    parsed_dataset_train = parsed_dataset_train.batch(BATCH_SIZE)
    parsed_dataset_train = parsed_dataset_train.prefetch(AUTO)
    
    # create generator
    iterator_train = iter(parsed_dataset_train)
     
    while True:
        try:
            parsed_record_train = iterator_train.get_next()
            #parsed_return = next(iter(parsed_record))
            parsed_return_train = mod_data(parsed_record_train, df_train)
            # yields training metadata and training image for model as well as target prediction
            yield [parsed_return_train['metadata'], parsed_return_train['image']], parsed_return_train['target']
        except:
            break
        
# get the final validation dataset to be used for the model - has option to be used for training
def get_validation_dataset(file_paths, train=False, ordered=False, validating=False):
    parsed_dataset_val = load_dataset(file_paths, ordered=ordered)
    parsed_dataset_val = parsed_dataset_val.cache()
    
    if validating:
        parsed_dataset_val = parsed_dataset_val.repeat()
    
    if train:
        parsed_dataset_val = parsed_dataset_val.repeat()
        parsed_dataset_val = parsed_dataset_val.map(augment_img, num_parallel_calls=AUTO)
        parsed_dataset_val = parsed_dataset_val.shuffle(2048)
        
    parsed_dataset_val = parsed_dataset_val.batch(BATCH_SIZE)
    parsed_dataset_val = parsed_dataset_val.prefetch(AUTO)
    
    # create generator
    iterator_val = iter(parsed_dataset_val)
    
    while True:
        try: 
            parsed_record_val = iterator_val.get_next()
            parsed_return_val = mod_data(parsed_record_val, df_train)
            # yields training metadata and training image for model as well as target prediction 
            yield [parsed_return_val['metadata'], parsed_return_val['image']], parsed_return_val['target']
        except:
            break
        
# get the test dataset to predict values after model has been trained
def get_test_dataset(file_paths, ordered=False):
    parsed_dataset_test = load_dataset(file_paths, labeled=False, ordered=ordered)
    parsed_dataset_test = parsed_dataset_test.batch(BATCH_SIZE)
    parsed_dataset_test = parsed_dataset_test.prefetch(AUTO)
    
    # create generator
    iterator_test = iter(parsed_dataset_test)
    
    while True:
        try: 
            parsed_record_test = iterator_test.get_next()
            parsed_return_test = mod_data(parsed_record_test, df_test)
            # yields test metadata and test image for model to use to predict the target
            yield [parsed_return_test['metadata'], parsed_return_test['image']], parsed_return_test['image_name']
        except:
            break


NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print(NUM_TRAINING_IMAGES)


# In[]:


# import model
model = load_model('../input/melanoma-classification-merged/merged_model.h5')


# In[]:


# load test files into memory
test_data = [(x, y) for x, y in get_test_dataset(TEST_FILENAMES)]

# predict probabilities
test_probabilities = model.predict(x for x, y in test_data)
#print(test_probabilities.flatten())

# image names
test_imgs = [y for x, y in test_data]
#print(test_imgs)


# In[]:


# reshape data
test_imgs_temp = []
for x in test_imgs:
    test_imgs_temp.append(x.numpy().astype('str').ravel())
test_imgs = np.concatenate(np.asarray(test_imgs_temp)).ravel().reshape((test_probabilities.shape[0], 1))


# In[]:


# create pandas dataframe and export to csv
df = pd.DataFrame({
    'image_name': test_imgs[:, 0],
    'target': test_probabilities[:, 0]
})
df.to_csv('submission.csv', index=False)

