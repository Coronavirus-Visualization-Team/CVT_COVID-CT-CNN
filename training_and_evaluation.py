import os, shutil, PIL, keras
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
import random
import numpy as np
import cv2

from pathlib import Path
from keras import layers, models, optimizers, regularizers

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.nasnet import NASNetMobile
from keras.applications import InceptionResNetV2
from keras.models import Model, Input, Sequential, load_model
from keras.layers import AveragePooling2D, Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.callbacks import Callback, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn import metrics

import matplotlib.image as mpimg

import itertools
from sklearn import metrics

import sklearn.metrics
import math
import pandas as pd

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

IMAGE_SIZE = (224, 224)
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
NUM_EPOCHS = 35
# percent of layers that will be trainable from original model
PERCENT_TRAINABLE = 0.5

TRAIN_DIR = '/content/drive/My Drive/CVT/data/train'
VAL_DIR = '/content/drive/My Drive/CVT/data/validation'
TEST_DIR = '/content/drive/My Drive/CVT/data/test'
# hold-out set
TEST2_DIR = '/content/drive/My Drive/CVT/HO-data'

NUM_TRAIN_SAMPLES = len(os.listdir(TRAIN_DIR + "/COVID")) + len(os.listdir(TRAIN_DIR + "/NonCOVID"))
NUM_VAL_SAMPLES = len(os.listdir(VAL_DIR + "/COVID")) + len(os.listdir(VAL_DIR + "/NonCOVID"))


def preprocess_data():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        brightness_range=[0.5, 1.0],
        shear_range=0.05,
        zoom_range=0.15,
        horizontal_flip=True,
        # vertical_flip=True,
        # fill_mode="constant",
        # cval=255
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        shuffle=False,
        class_mode='binary'
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=TRAIN_BATCH_SIZE,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=VAL_BATCH_SIZE,
        class_mode='binary'
    )

    test_generator2 = test_datagen.flow_from_directory(
        TEST2_DIR,
        target_size=IMAGE_SIZE,
        shuffle=False,
        class_mode='binary'
    )

    return (train_generator, validation_generator, test_generator, test_generator2)


def add_noise_contrast(img):
    VARIABILITY = 25
    deviation = VARIABILITY * random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img


def augment(img):
    if np.random.random() < 0.3:
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
    if np.random.random() < 0.3:
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        # return add_noise_contrast(img)
    return img


def test_augmentation():
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.04,
        height_shift_range=0.04,
        brightness_range=[0.4, 1.0],
        shear_range=0.05,
        zoom_range=0.2,
        horizontal_flip=True,
        # vertical_flip=True,
        # fill_mode="constant",
        # cval=255

    )

    fnames_covid = [os.path.join(TRAIN_DIR, 'COVID', fname) for fname in os.listdir(os.path.join(TRAIN_DIR, 'COVID'))]
    fnames_noncovid = [os.path.join(TRAIN_DIR, 'NonCOVID', fname) for fname in
                       os.listdir(os.path.join(TRAIN_DIR, 'NonCOVID'))]
    fnames = fnames_covid + fnames_noncovid
    for r in range(3):
        img_path = fnames[random.randint(0, len(fnames) - 1)]
        print(img_path)

        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        x = image.img_to_array(img)
        x = x.reshape((1,) + x.shape)
        print(x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1):
            plt.figure(i)
            imgplot = plt.imshow(image.array_to_img(batch[0]))
            i += 1
            if (i % 3 == 0):
                break
        plt.show()


def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def f2(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f2_val = (5 * precision * recall) / (4 * precision + recall + K.epsilon())
    return f2_val


def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def balanced_acc(y_true, y_pred):
    return (sensitivity(y_true, y_pred) + specificity(y_true, y_pred)) / 2


def build_transfer():
    IMG_SHAPE = (224, 224, 3)

    for index, layer in enumerate(TRANSFER_MODEL.layers):
        if index < len(TRANSFER_MODEL.layers) * PERCENT_TRAINABLE:
            layer.trainable = False
        else:
            layer.trainable = True

    X = layers.Flatten()(TRANSFER_MODEL.output)
    X = layers.Dense(64)(X)
    X = layers.Activation('relu')(X)
    X = layers.Dropout(0.5)(X)
    X = layers.Dense(1, activation='sigmoid')(X)

    model = Model(TRANSFER_MODEL.input, X)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[
                      tf.keras.metrics.BinaryAccuracy(threshold=0.5),
                      balanced_acc,
                      f1,
                      f2,
                      tf.keras.metrics.Recall(name="recall"),
                      specificity,
                      tf.keras.metrics.Precision(name='precision'),
                      tf.keras.metrics.AUC(name="auc")])
    return model


def graph_metrics(history):
    metrics_list = list(history.history.keys())
    num_metrics = int(len(metrics_list) / 2)

    fig, axs = plt.subplots(math.ceil(num_metrics / 2), 2, figsize=(20, 35))

    epochs = range(1, NUM_EPOCHS + 1)

    for i in range(num_metrics):
        train_metric = history.history[metrics_list[i]]
        val_metric = history.history[metrics_list[i + num_metrics]]
        row = int(i / 2)
        col = 1 * (i % 2 != 0)
        axs[row, col].plot(epochs, train_metric, label='Training ' + metrics_list[i])
        axs[row, col].plot(epochs, val_metric, label='Validation' + metrics_list[i])
        axs[row, col].set_title(metrics_list[i])
        axs[row, col].legend()
    plt.show()


def train(model, train_generator, validation_generator):
    history = model.fit(
        train_generator,
        steps_per_epoch=NUM_TRAIN_SAMPLES // TRAIN_BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=NUM_VAL_SAMPLES // VAL_BATCH_SIZE,
        callbacks=[tf.keras.callbacks.ReduceLROnPlateau(patience=5)]
    )
    return history


def run(model=DenseNet121(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), include_top=False)):
    model = build_transfer()
    train_generator, validation_generator, test_generator, test2_generator = preprocess_data()
    history = train(model, train_generator, validation_generator)
    model.save("my_h5_model.h5")
    return model






