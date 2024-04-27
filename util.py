import numpy as np
import tensorflow as tf
import scipy
from scipy import sparse, stats, linalg

import os
import userinput


def get_sparse_random_matrix(mat_size):
    temp_dia = np.diag(np.random.rand(mat_size))
    temp_sprs = sparse.random(mat_size, mat_size, density=0.2).toarray()

    K_mats = temp_sprs + temp_dia
    K_mats = np.array([K_mats], dtype=float)

    return K_mats


def generate_matrix(inp):
    os.makedirs(inp.data_path, exist_ok=True)
    os.chdir(inp.data_path)

    mat_size = inp.matrix_size
    n_mat = inp.n_matrix

    #numpy random matrix
    # K_mats = np.array([np.random.randn(mat_size, mat_size)], dtype=float)
    # K_invs = np.array([np.linalg.inv(K_mats[0])])
    #
    # for i in range(n_mat-1):
    #     new_mat = np.random.randn(mat_size, mat_size)
    #     if np.linalg.det(new_mat) != 0:
    #         inv_mat = np.linalg.inv(new_mat)
    #         K_mats = np.insert(K_mats, [0], new_mat, axis=0)
    #         K_invs = np.insert(K_invs, [0], inv_mat, axis=0)

    # sparse random matrix
    K_mats = get_sparse_random_matrix(mat_size)
    K_invs = np.array([np.linalg.inv(K_mats[0])])

    for i in range(n_mat - 1):
        new_mat = get_sparse_random_matrix(mat_size)
        inv_mat = np.linalg.inv(new_mat)
        K_mats = np.insert(K_mats, [0], new_mat, axis=0)
        K_invs = np.insert(K_invs, [0], inv_mat, axis=0)

    str_msz = str(mat_size)
    filename_K_mats = "K_mats_" + str_msz + "by"+ str_msz + "(" + str(n_mat) + ")"
    filename_K_invs = "K_invs_" + str_msz + "by"+ str_msz + "(" + str(n_mat) + ")"

    np.save(filename_K_mats, K_mats)
    np.save(filename_K_invs, K_invs)

    return K_mats, K_invs


def load_matrix_from_file(filename):
    # os.chdir(DATA_DIR)
    my_mat = np.load(filename)

    return my_mat


def floss(y_true, y_pred):
    # loss = || I - AA^{-1}||
    # y_true is the true inverse
    # y_pred is the predicted inverse
    # A is the original not inverted matrix
    # A^{-1} is the inverted matrix
    # I is the identiy matrix
    """
    Iterate trough tensor elements in y_true and y_pred
    WARNING: SLOW!!
    def single_floss(elems):
        eye = tf.eye(self.msize)
        return tf.norm(tf.subtract(eye, tf.linalg.matmul(tf.linalg.inv(elems[0]), elems[1])), ord='euclidean')
    elems_ = (y_true, y_pred)
    return tf.reduce_mean(tf.map_fn(single_floss, elems_, dtype=tf.float32))
    """

    """
    Iterate trough matrix and calculate the norm of each matrix in a tensor
    WARNING: SLOW!!
    def apply_norm(elem):
        return tf.norm(elem)
    eye = tf.eye(self.msize,
                 batch_shape=[tf.shape(y_true)[0]])
    # return tf.norm(tf.subtract(eye, tf.linalg.matmul(tf.linalg.inv(y_true), y_pred)), ord='euclidean')
    elems = tf.subtract(eye, tf.linalg.matmul(tf.linalg.inv(y_true), y_pred))
    norms = tf.map_fn(apply_norm, elems, dtype=tf.float32)
    """

    """
    Fast Forbenius L-2 Norm
    """
    eye = tf.eye(28,
                 batch_shape=[tf.shape(y_true)[0]])
    res = tf.subtract(eye, tf.linalg.matmul(tf.linalg.inv(y_true), y_pred))
    res = tf.abs(res)
    res = tf.square(res)
    res = tf.reduce_sum(tf.reduce_sum(res, axis=1), axis=1)
    res = tf.sqrt(res)
    res = tf.reduce_mean(res)
    return res


def custom_loss_inverse_mat(y_true, y_pred):
    eye = tf.eye(tf.shape(y_true[0])[0], batch_shape=[tf.shape(y_true)[0]])
    eye_pred = tf.linalg.matmul(tf.linalg.inv(y_true), y_pred)

    eye = tf.cast(eye, tf.float32)
    eye_pred = tf.cast(eye_pred, tf.float32)

    res = tf.subtract(eye, eye_pred)
    res = tf.square(res)
    res = tf.reduce_mean(res)

    return res

    # eye = tf.eye(28, batch_shape=[tf.shape(y_true)[0]])
    #
    # res = tf.subtract(eye, tf.linalg.matmul(tf.linalg.inv(y_true), y_pred))
    # res = tf.abs(res)
    # res = tf.square(res)
    # res = tf.reduce_sum(tf.reduce_sum(res, axis=1), axis=1)
    # res = tf.sqrt(res)
    # res = tf.reduce_mean(res)
    # return res

    # loss = ||I - AA(-1)||
    # y_true is the true inverse
    # y_pred is the predicted inverse
    # A is the original not inverted matrix
    # A^{-1} is the inverted matrix
    # I is the identiy matrix
    """
    Iterate trough tensor elements in y_true and y_pred
    WARNING: SLOW!!
    def single_floss(elems):
        eye = tf.eye(self.msize)
        return tf.norm(tf.subtract(eye, tf.linalg.matmul(tf.linalg.inv(elems[0]), elems[1])), ord='euclidean')
    elems_ = (y_true, y_pred)
    return tf.reduce_mean(tf.map_fn(single_floss, elems_, dtype=tf.float32))
    """


if __name__ in "__main__":
    y_true = [[[-0.1,-0.1,-0.7],
               [0.4,0.3,-0.6],
               [0.8,-0.5,0.1]],
              [[0.4,-1.0,0.4],
               [-0.3,-0.6,-0.2],
               [-0.8,0.4,0.5]]]

    y_pred = [[[-0.2,-0.2,-0.7],
               [0.4,0.3,-0.7],
               [0.9,-0.5,0.1]],
              [[0.3,-0.8,0.4],
               [-0.4,-0.6,-0.1],
               [-0.7,0.3,0.6]]]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    custom_loss_inverse_mat(y_true, y_pred)
