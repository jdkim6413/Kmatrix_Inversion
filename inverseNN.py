import numpy as np
import random
from keras.layers import *
from keras import models, Sequential, metrics
from tensorflow import keras
import tensorflow as tf
import util
import userinput

import os
import json

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from sklearn.model_selection import train_test_split

# 시드 고정
SEED = 99
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


class InverseNN:
    def __init__(self, inp, K_mats, K_invs):
        self.K_mats = K_mats
        self.K_invs = K_invs

        self.mat_size = K_mats[0].shape[0]
        self.n_matrix = K_mats.shape[0]

        self.L_mats = None
        self.U_mats = None

        mat_fname = "K_mats_" + str(self.mat_size) + "by" + str(self.mat_size) + "(" + str(self.n_matrix) + ")"
        inv_fname = "K_invs_" + str(self.mat_size) + "by" + str(self.mat_size) + "(" + str(self.n_matrix) + ")"

        self.mat_name = mat_fname
        self.inv_name = inv_fname

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

        # self.optimizer = keras.optimizers.SGD(learning_rate=0.001)
        self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
        # self.loss_fn = keras.losses.MeanSquaredError()
        self.loss_fn = util.custom_loss_inverse_mat
        self.metrics = [metrics.MeanSquaredError(), metrics.MeanAbsoluteError()]

        self.model = None
        self.history = None

        self.data_path = inp.data_path

    def create_model(self, epochs=2, batch_size=64, is_fromfile=False):
        if not is_fromfile:
            self._define_dataset()

            opt = self.optimizer
            loss_fn = self.loss_fn
            my_metrics = self.metrics

            model = self._model_2(opt, loss_fn, my_metrics)
            history = model.fit(self.X_train, self.y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(self.X_val, self.y_val))

            self.history = history.history
            self.model = model

            #temporary_test_code
            test_mat = util.get_sparse_random_matrix(self.mat_size)

            for i in range(100 - 1):
                new_mat = util.get_sparse_random_matrix(self.mat_size)
                test_mat = np.insert(test_mat, [0], new_mat, axis=0)

            pred_mat = model.predict(test_mat)

            test_mat_inv = np.linalg.inv(test_mat)
            mul_test_pred = np.matmul(test_mat, pred_mat)

            import pandas as pd

            def tmp_fn(arr, name):
                arr_reshaped = arr.reshape(arr.shape[0], -1)
                np.savetxt(name, arr_reshaped, delimiter=",")

            tmp_fn(test_mat, "test_mat.csv")
            tmp_fn(pred_mat, "pred_mat.csv")
            tmp_fn(test_mat_inv, "test_mat_inv.csv")
            tmp_fn(mul_test_pred, "mul_test_pred.csv")

            # print(test_mat)
            # print(pred_mat)
            # print(np.linalg.inv(test_mat))
            #
            # print(np.matmul(test_mat, pred_mat))

            # print(test_mat)
            # print(pred_mat)
            # print(test_mat - pred_mat)

            self._save_model()
        else:
            self._load_model()

    def eval_model(self):
        model = self.model
        # model.evaluate(
        #     x=None,
        #     y=None,
        #     batch_size=None,
        #     verbose="auto",
        #     sample_weight=None,
        #     steps=None,
        #     callbacks=None,
        #     return_dict=False
        # )

    def _define_dataset(self):
        input_mat_x = self.K_mats
        # output_mat_y = self.K_mats
        output_mat_y = self.K_invs

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(input_mat_x,
                                                                              output_mat_y,
                                                                              test_size=0.2,
                                                                              random_state=99)

    # def _conv_autoencoder_model(self, n_filter, dense1, dense2, dense3, lr=1.2e-5):

    def _model_1(self, optmizer, loss_fn, metrics):
        x_tr = self.X_train
        x_val = self.X_val
        y_tr = self.y_train
        y_val = self.y_val

        self.X_train = np.reshape(x_tr, (x_tr.shape[0], self.mat_size, self.mat_size, 1))
        self.X_val = np.reshape(x_val, (x_val.shape[0], self.mat_size, self.mat_size, 1))
        self.y_train = np.reshape(y_tr, (y_tr.shape[0], self.mat_size, self.mat_size, 1))
        self.y_val = np.reshape(y_val, (y_val.shape[0], self.mat_size, self.mat_size, 1))

        keras.backend.clear_session()
        tf.random.set_seed(SEED)

        model = Sequential()

        # encoder network
        model.add(Conv2D(filters=128, kernel_size=(2, 2), activation='relu', padding='same',
                         input_shape=(self.mat_size, self.mat_size, 1)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(2, 2), activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(2, 2), activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Conv2D(filters=512, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding='same'))

        # decoder network
        model.add(Conv2D(filters=512, kernel_size=(2, 2), activation='relu', padding='same'))

        model.add(
            tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=(2, 2), strides=(2, 2), activation='relu',
                                            padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(2, 2), activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(2, 2), activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(2, 2), activation='relu', padding='same'))

        model.add(
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), activation='relu',
                                            padding='same'))
        model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu', padding='same'))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(Conv2D(filters=1, kernel_size=(2, 2), activation='relu', padding='same'))

        # summary of the model
        model.summary()

        # compile model
        model.compile(
            optimizer=optmizer,
            loss=loss_fn,
            metrics=metrics
        )

        return model

    def _model_2(self, optmizer, loss_fn, metrics):
        keras.backend.clear_session()

        msize = self.mat_size
        nunits = msize * msize

        model = Sequential()
        model.add(Flatten(input_shape=(msize, msize, )))

        model.add(Dense(9, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(9, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(9, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(msize * msize))
        model.add(Reshape((msize, msize)))

        # summary of the model
        model.summary()

        # compile model
        model.compile(
            optimizer=optmizer,
            loss=loss_fn,
            metrics=metrics
        )

        return model

    def _save_model(self):
        os.makedirs(self.data_path, exist_ok=True)
        os.chdir(self.data_path)
        model = self.model

        # save model
        model.save("current_model")

        # save history
        json.dump(self.history, open('train_history.json', 'w'))
        self._plot_history()
        print('\n Model Saved')

    def _load_model(self):
        os.chdir(self.model_dir)

        # load model
        loaded_model = models.load_model("current_model")
        self.model = loaded_model

        # load history
        self.history = json.load(open('train_history.json', 'r'))
        print('\n Model Loaded')

    def _plot_history(self, is_plot=False):
        history = self.history

        print('\n-------- Plotting Process (Accuracy & Loss) --------')
        plt.rcParams["figure.figsize"] = (12, 4)
        plt.rcParams.update({'font.size': 12})
        plt.figure()
        plt.style.use('default')
        plt.subplot(121)
        plt.title('Loss Plot')
        plt.plot(history['loss'], label='train loss')
        plt.plot(history['val_loss'], label='val loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.subplot(122)
        plt.title('MSE Plot')
        plt.plot(history['mean_squared_error'], label='train MSE')
        plt.plot(history['val_mean_squared_error'], label='val MSE')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig('Training_history.png', dpi=360)
        if is_plot:
            plt.show()
        plt.clf()
        plt.close('all')
