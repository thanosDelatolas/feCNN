from os import name
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import (Dense)
from copy import deepcopy

import datetime

import losses

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
        self.n_elec = self.leadfield.shape[0]
        self.n_dipoles = self.leadfield.shape[1]

        self.sim = deepcopy(sim)

        # simulation's samples
        self.n_samples = self.sim.eeg_data.shape[1]
        self.activation_function = activation_function
        self.n_jobs = n_jobs
        self.compiled = False

        self.default_loss = losses.weighted_huber_loss

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

            # add input layer
            self.model.add(keras.Input(shape=(self.n_elec,), name='Input'))
            #  Number of neurons per hidden layer.
            n_neurons = 100
            self.model.add(Dense(units=n_neurons, activation=self.activation_function, name='Hidden'))

            # Add output layer
            self.model.add(Dense(self.n_dipoles, activation='linear', name='Output'))

            self.model.summary()

    def fit(self, learning_rate=0.001, 
        validation_split=0.1, epochs=50, metrics=None, 
        false_positive_penalty=2, delta=1., batch_size=128, 
        loss=None, patience=7  
    ):
        ''' Train the neural network using training data (eeg) and labels (sources).

        The training data are stored in the simulation object
        
        Parameters
        ----------

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
        false_positive_penalty : float
            Defines weighting of false-positive predictions. Increase for conservative 
            inverse solutions, decrease for liberal prediction.
        batch_size : int
            The number of samples to simultaneously calculate the error 
            during backpropagation.
        loss : tf.keras.losses
            The loss function.
        Return
        ------
        self : esinet.Net
            Method returns the object itself.

        '''

        if len(self.sim.eeg_data.shape) != 2 :
            raise AttributeError("EEG data must be 2D (n_elctrodes x n_samples")
        elif len(self.sim.source_data.shape) != 2 :
            raise AttributeError("Sources data must be 2D (n_dipoles x n_samples")

        # Input data
        x = self.sim.eeg_data.T

        # Target data
        y = self.sim.source_data.T

        # early stoping
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', \
            mode='min', patience=patience, restore_best_weights=True)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


        if loss == None:
            loss = self.default_loss(weight=false_positive_penalty, delta=delta)
        
        if metrics is None:
            metrics = [self.default_loss(weight=false_positive_penalty, delta=delta)]
        
        if not self.compiled:
            self.model.compile(optimizer, loss, metrics=metrics)
            self.compiled = True
        
       
        callbacks = [es]

        self.model.fit(x, y, 
                epochs=epochs, batch_size=batch_size, shuffle=True, 
                validation_split=validation_split, callbacks=callbacks)
        return self
