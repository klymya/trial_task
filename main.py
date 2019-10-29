import os
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint


logging.basicConfig(level=logging.INFO)


BEST_WEIGHTS_PATH = 'best_weights.h5'


def fix_object_to_float_nan(df):
    """
    convert columns with object dtype to float, because nan in data
    :param df: pandas DataFrame
    :return: dataframe with fixed columns' dtype
    """
    cols = df.select_dtypes(include=['object']).columns
    df[cols] = df[cols].apply(pd.to_numeric, downcast='float', errors='coerce')
    return df


def fill_missing_values(df):
    """
    fill missing values with median for real and mode(s) for categorical
    :param df: pandas DataFrame
    :return: dataframe without missing values
    """
    df = df.copy()

    cols = df.select_dtypes(include=['float64', 'float32']).columns
    df[cols] = df[cols].fillna(df[cols].median())

    cols = df.select_dtypes(include=['int64']).columns
    df[cols] = df[cols].fillna(df[cols].mode())

    return df


def f1(y_true, y_pred):
    """
    Macro averaged F1 score for Keras models
    :param y_true: labels
    :param y_pred: prediction
    :return: macro f1 score
    """
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def build_model(input_shape, output_shape=3):
    """
    build not deep neural network
    :param input_shape: input data dimention
    :param output_shape: number of classes, dimention of output layer
    :return: Keras neural network
    """
    logging.info('model building')
    model = Sequential()

    model.add(
        Dense(units=100, input_dim=input_shape, activation='tanh'))
    model.add(BatchNormalization(trainable=True))
    model.add(Dropout(0.5))

    model.add(Dense(units=100, activation='tanh'))
    model.add(BatchNormalization(trainable=True))
    model.add(Dropout(0.5))

    model.add(Dense(units=100, activation='tanh'))
    model.add(BatchNormalization(trainable=True))
    model.add(Dropout(0.5))

    model.add(Dense(units=100, activation='tanh'))
    model.add(BatchNormalization(trainable=True))
    model.add(Dropout(0.5))

    model.add(Dense(units=output_shape, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adam",
                  metrics=[f1])

    logging.info(model.summary())
    return model


def run_pipeline(train_path, test_path, prediction_path):
    """
    read data, pre-process them, train a model and save prediction to file
    :param train_path: path to training data csv
    :param test_path: path to test data csv
    :param prediction_path: path where prediction should be saved
    """
    logging.info('data reading and pre-processing')
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    columns_to_remove = ['dec', 'ra', 'objid', 'class']
    train_y = train_df['class']
    test_objid = test_df['objid']

    train_df.drop(columns_to_remove, axis=1, inplace=True, errors='ignore')
    test_df.drop(columns_to_remove, axis=1, inplace=True, errors='ignore')

    train_df = fix_object_to_float_nan(train_df)
    test_df = fix_object_to_float_nan(test_df)

    train_df = fill_missing_values(train_df)
    test_df = fill_missing_values(test_df)

    scaler = StandardScaler().fit(train_df)
    scaled_train_df = scaler.transform(train_df)
    scaled_test_df = scaler.transform(test_df)

    model = build_model(scaled_train_df.shape[1])

    logging.info('training')
    early_stopping = EarlyStopping(monitor='val_f1', patience=15, verbose=1,
                                   mode='max')
    mcp_save = ModelCheckpoint(BEST_WEIGHTS_PATH, save_best_only=True,
                               monitor='val_f1', mode='max', verbose=1)

    model.fit(scaled_train_df, to_categorical(train_y),
              validation_split=0.15,
              batch_size=128, epochs=200,
              verbose=True, callbacks=[early_stopping, mcp_save])

    logging.info('prediction')
    model.load_weights(BEST_WEIGHTS_PATH)
    pred = model.predict(scaled_test_df)
    pred = np.argmax(pred, axis=1)

    pd.DataFrame({'objid': test_objid, 'prediction': pred})\
        .to_csv(prediction_path)

    os.remove(BEST_WEIGHTS_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True,
                        help='path to csv file with training data')
    parser.add_argument('--unlabeled_path', type=str,
                        help='path to csv file with unlabeled data')
    parser.add_argument('--test_path', type=str, required=True,
                        help='path to csv file with test data')
    parser.add_argument('--prediction_path', type=str, required=True,
                        help='path to csv file where prediction will be saved')
    args = parser.parse_args()

    run_pipeline(args.train_path, args.test_path, args.prediction_path)
