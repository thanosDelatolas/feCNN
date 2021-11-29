import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import (Dense)
from copy import deepcopy

class EEGNet:
    ''' The neural network class that creates and trains the model. 
    
    Attributes
    ----------
    fwd : The forward object located in forward.py
    sim : The simulation object located in simulation.py
                 
    activation_function : str
        The activation function used for each fully connected layer.
    n_jobs : int
        Number of jobs/ cores to use during parallel processing

    Methods
    -------
    fit : trains the neural network with the EEG and source data
    train : trains the neural network with the EEG and source data
    predict : perform prediciton on EEG data
    evaluate : evaluate the performance of the model
    '''

    def __init__(self, fwd, sim, activation_function='swish', n_jobs=-1):
        self.leadfield = deepcopy(fwd.leadfield)
        self.n_channels = self.leadfield.shape[0]
        self.n_dipoles = self.leadfield.shape[1]

        self.sim = deepcopy(sim)

        self.activation_function = activation_function
        self.n_jobs = n_jobs
        self.compiled = False

        if self.leadfield.shape[0] != self.sim.eeg_data.shape[0] or self.leadfield.shape[1] != self.sim.source_data.shape[0] :
            msg = 'Incompatible sim and fwd objects'
            raise AttributeError(msg)
    
    def build_model(self):
        ''' Build the neural network architecture using the 
        tensorflow.keras.Sequential() API. 

        The architecture is  a simple single hidden layer fully connected ANN for single time instance data.
       
        '''
        if not self.compiled :
            # Build the artificial neural network model using Dense layers.
            self.model = keras.Sequential()

            #  Number of neurons per hidden layer.
            n_neurons = 100
            self.model.add(Dense(units=n_neurons, activation=self.activation_function))

            # Add output layer
            self.model.add(Dense(self.n_dipoles, activation='linear'))

            # Build model with input layer
            self.model.build(input_shape=(None, self.n_channels))

            self.model.summary()

    def fit(self, optimizer=None, learning_rate=0.001, 
        validation_split=0.1, epochs=50, metrics=None, device=None, 
        false_positive_penalty=2, delta=1., batch_size=128, loss=None, 
        sample_weight=None, return_history=False, dropout=0.2, patience=7, 
        tensorboard=False):
        ''' Train the neural network using training data (eeg) and labels (sources).

        The training data are stored in the simulation object
        
        Parameters
        ----------
        
        optimizer : tf.keras.optimizers
            The optimizer that for backpropagation.
        learning_rate : float
            The learning rate for training the neural network
        validation_split : float
            Proportion of data to keep as validation set.
        delta : int/float
            The delta parameter of the huber loss function
        epochs : int
            Number of epochs to train. In one epoch all training samples 
            are used once for training.
        metrics : list/str
            The metrics to be used for performance monitoring during training.
        device : str
            The device to use, e.g. a graphics card.
        false_positive_penalty : float
            Defines weighting of false-positive predictions. Increase for conservative 
            inverse solutions, decrease for liberal prediction.
        batch_size : int
            The number of samples to simultaneously calculate the error 
            during backpropagation.
        loss : tf.keras.losses
            The loss function.
        sample_weight : numpy.ndarray
            Optional numpy array of sample weights.

        Return
        ------
        self : esinet.Net
            Method returns the object itself.

        '''

        if self.sim.eeg_data.shape != 2 :
            raise AttributeError("EEG data must be 2D (n_elctrodes x n_samples")
        elif self.sim.sources_data.shape != 2 :
            raise AttributeError("Sources data must be 2D (n_dipoles x n_samples")

        # Input data
        x = self.sim.eeg_data

        # Target data
        y = self.sim.source_data

        # early stoping
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', \
            mode='min', patience=patience, restore_best_weights=True)
        