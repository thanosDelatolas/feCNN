import sys
import os

import scipy.io

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import duneuropy as dp

folder_input = os.path.join(parent,'duneuropy/Data')
dipoles_filename = os.path.join(folder_input, 'dipoles.mat')

dipoles =  scipy.io.loadmat(dipoles_filename)['cd_matrix']

print(dipoles.shape)
