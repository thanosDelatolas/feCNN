import numpy as np
from copy import deepcopy
import scipy.io
import pickle

    
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def rms(x):
    ''' Calculate the root mean square of some signal x.
    Parameters
    ----------
    x : numpy.ndarray, list
        The signal/data.

    Return
    ------
    rms : float
    '''
    return np.sqrt(np.mean(np.square(x)))

def repeat_newcol(x, n):
    ''' Repeat a list/numpy.ndarray x in n columns.'''
    out = np.zeros((len(x), n))
    for i in range(n):
        out[:,  i] = x
    return np.squeeze(out)


def get_n_order_indices(order, pick_idx, neighbors):
    ''' Iteratively performs region growing by selecting neighbors of 
    neighbors for <order> iterations.
    '''
    current_indices = np.array([pick_idx])

    if order == 0:
        return current_indices

    for _ in range(order):
        current_indices = np.append(current_indices, np.concatenate(neighbors[current_indices]))

    return np.unique(np.array(current_indices))

def gaussian(x, mu, sigma):
    ''' Gaussian distribution function.
    
    Parameters
    ----------
    x : numpy.ndarray, list
        The x-value.
    mu : float
        The mean of the gaussian kernel.
    sigma : float
        The standard deviation of the gaussian kernel.
    Return
    ------
    '''
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

def get_triangle_neighbors(tris_lr):
    if not np.all(np.unique(tris_lr[0]) == np.arange(len(np.unique(tris_lr[0])))):
        for hem in range(2):
            old_indices = np.sort(np.unique(tris_lr[hem]))
            new_indices = np.arange(len(old_indices))
            for old_idx, new_idx in zip(old_indices, new_indices):
                tris_lr[hem][tris_lr[hem] == old_idx] = new_idx

        # print('indices were weird - fixed them.')
    numberOfDipoles = len(np.unique(tris_lr[0])) + len(np.unique(tris_lr[1]))
    neighbors = [list() for _ in range(numberOfDipoles)]
    # correct right-hemisphere triangles
    tris_lr_adjusted = deepcopy(tris_lr)
    # the right hemisphere indices start at zero, we need to offset them to start where left hemisphere indices end.
    tris_lr_adjusted[1] += int(numberOfDipoles/2)
    # left and right hemisphere
    for hem in range(2):
        for idx in range(numberOfDipoles):
            # Find the indices of the triangles where our current dipole idx is part of
            trianglesOfIndex = tris_lr_adjusted[hem][np.where(tris_lr_adjusted[hem] == idx)[0], :]
            for tri in trianglesOfIndex:
                neighbors[idx].extend(tri)
                # Remove self-index (otherwise neighbors[idx] is its own neighbor)
                neighbors[idx] = list(filter(lambda a: a != idx, neighbors[idx]))
            # Remove duplicates
            neighbors[idx] = list(np.unique(neighbors[idx]))
    return neighbors


def read_electrodes(filename):
    ''' Reads the electrodes positions from a .elc file
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
    startPositions = next(i for i,l in enumerate(lines) if l.startswith('Positions'))+1
    endPositions = next(i for i,l in enumerate(lines) if l.startswith('NumberPolygons'))
    positionLines = lines[startPositions:endPositions]
    electrodePositions = [[float(y) for y in x.strip().split()] for x in positionLines]
    return electrodePositions

def read_dipoles(filename):
    dipoles =  scipy.io.loadmat(filename)['cd_matrix']

    dipPos = list()
    dipMom = list()
    for i in range(len(dipoles)):
        dipPos.append(dipoles[i][:3].tolist())
        dipMom.append(dipoles[i][3:].tolist())
    
    return dipPos, dipMom

def save_object(obj, filename):
    ''' Saves a python object to file
    '''
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    ''' Load an object which is saved as pickle.
    '''
    with open(filename, 'rb') as outp:  
        obj = pickle.load(outp)
    return obj


def scale_array(array):
        ''' Scales an array.
        Usually, prior to training the neural network
        '''
        return array/np.max(array)