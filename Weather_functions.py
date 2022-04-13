# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 08:40:01 2022

@author: ramch
"""

#https://towardsdatascience.com/deep-transfer-learning-for-image-classification-f3c7e0ec1a14

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.python.keras import layers, optimizers
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

#https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
#https://keras.io/api/preprocessing/image/   
def augment(pre,train,test):
    train_augment = ImageDataGenerator(preprocessing_function=pre, validation_split=0.2)
    test_augment = ImageDataGenerator(preprocessing_function=pre)
    
    train_gen = train_augment.flow_from_dataframe(
        dataframe=train,
        x_col='File_Path',
        y_col='Labels',
        target_size=(224,224),
        class_mode='categorical',
        batch_size=64,
        shuffle=True,
        seed=0,
        subset='training',
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    valid_gen = train_augment.flow_from_dataframe(
        dataframe=train,
        x_col='File_Path',
        y_col='Labels',
        target_size=(224,224),
        class_mode='categorical',
        batch_size=32,
        shuffle=False,
        seed=0,
        subset='validation',
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    test_gen = test_augment.flow_from_dataframe(
        dataframe=test,
        x_col='File_Path',
        y_col='Labels',
        target_size=(224,224),
        color_mode='rgb',
        seed=0,
        class_mode='categorical',
        batch_size=64,
        verbose=0,
        shuffle=False)
    return train_gen, valid_gen, test_gen

def run_model(mod_name):
    pre_model = mod_name(input_shape=(224,224, 3),
                   include_top=False,
                   weights='imagenet',
                   pooling='avg')
    #lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #initial_learning_rate=1e-2,
    # decay_steps=10000,
    # decay_rate=0.9)
    pre_model.trainable = False
    inputs = pre_model.input
    x = Dense(100, activation='relu')(pre_model.output)
    x = Dense(100, activation='relu')(x)
    outputs = Dense(11, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    #model.compile(loss = 'categorical_crossentropy',optimizer=keras.optimizers.SGD(learning_rate=lr_schedule),metrics=['accuracy'])
    model.compile(loss = 'categorical_crossentropy',optimizer=tensorflow.optimizers.Adam(5e-5),metrics=['accuracy'])
    early_stop  = [EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=3,
                              mode='min')]
    return model, early_stop
#https://keras.io/api/callbacks/early_stopping/

def ploting(history,test_gen,train_gen,model,testLabel,testFilePath):
    # Plotting Accuracy, val_accuracy, loss, val_loss
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax = ax.ravel()

    for i, parameter in enumerate(['accuracy', 'loss']):
        ax[i].plot(history.history[parameter])
        ax[i].plot(history.history['val_' + parameter])
        ax[i].set_title(f'Model {parameter}')
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(parameter)
        ax[i].legend(['Train', 'Validation'])
        
    # Predict Data Test
    pred = model.predict(test_gen )
    pred = np.argmax(pred,axis=1)
    labels = (train_gen.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    pred = [labels[k] for k in pred]
    
    # Classification report
    cm=confusion_matrix(testLabel,pred)
    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(cm,
                     index = ['dew', 'fogsmog','frost','glaze','hail','lightning','rain','rainbow','rime','sandstorm','snow'], 
                     columns = ['dew', 'fogsmog','frost','glaze','hail','lightning','rain','rainbow','rime','sandstorm','snow'])
    
    #Plotting the confusion matrix
    plt.figure(figsize=(20,15))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
    clr = classification_report(testLabel, pred)
    #print(cm)
    print(clr)
    # Display 3 picture of the dataset with their labels
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 8),
                        subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(testFilePath.iloc[i+1]))
        ax.set_title(f"Actual: {testLabel.iloc[i+1]}\nPredicted: {pred[i+1]}")
    plt.tight_layout()
    plt.show()
        
    return history

def result_test(test,trained_model):
    results = trained_model.evaluate(test, verbose=1)
    
    print("    Test Loss: {:.5f}".format(results[0]))
    print("Test Accuracy: {:.2f}%".format(results[1] * 100))
    
    return results


#fine tuning
#https://www.tensorflow.org/tutorials/images/transfer_learning
