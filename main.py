# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import inverseNN
import util, userinput
from scipy import stats, sparse

import os

if __name__ == '__main__':
    # inputs
    gen_mat = True
    mat_size = 3
    n_mat = 10000

    inp = userinput.UserInput(gen_matrix=gen_mat, matrix_size=mat_size, n_matrix=n_mat)
    os.chdir(inp.drive_path)

    epochs = 10
    batch_size = 32

    if inp.gen_matrix:
        K_mats, K_invs = util.generate_matrix(inp)
    else:
        os.chdir(inp.data_path)
        K_mats = util.load_matrix_from_file('K_mats_3by3(50000).npy')
        K_invs = util.load_matrix_from_file('K_invs_3by3(50000).npy')

    my_NN = inverseNN.InverseNN(inp, K_mats, K_invs)
    my_NN.create_model(epochs, batch_size)




