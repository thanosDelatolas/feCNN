import mne
from copy import deepcopy
from . import simulation

from . import util
import numpy as np

class Net:
    ''' The neural network class that creates and trains the model.

        Attributes
        ----------
        fwd : mne.Forward
            the mne.Forward forward model class.
    '''

    def __init__(self, fwd):
        self.fwd = deepcopy(fwd)
        self._embed_fwd(fwd)

    def _embed_fwd(self, fwd):
        ''' Saves crucial attributes from the Forward model.
        
        Parameters
        ----------
        fwd : mne.Forward
            The forward model object.
        '''
        _, leadfield, _, _ = util.unpack_fwd(fwd)
        self.fwd = deepcopy(fwd)
        self.leadfield = leadfield
        self.n_channels = leadfield.shape[0]
        self.n_dipoles = leadfield.shape[1]

    
    def fit(self,simulation, optimizer=None, learning_rate=0.001, 
        validation_split=0.1, epochs=50, metrics=None, 
        false_positive_penalty=2, delta=1., batch_size=128, loss=None, 
        patience=7):
        ''' Train the neural network using training data (eeg) and labels (sources).
        
        Parameters
        ----------
        simulation : simulation.Simulation
            The Simulation object

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

        eeg = simulation.eeg_data
        sources = simulation.source_data
        self.subject = sources.subject if type(sources) == mne.SourceEstimate  else sources[0].subject
    
    def scale_eeg(self, eeg):
        ''' Scales the EEG prior to training/ predicting with the neural 
        network.

        Parameters
        ----------
        eeg : numpy.ndarray
            A 3D matrix of the EEG data (samples, channels, time_points)
        
        Return
        ------
        eeg : numpy.ndarray
            Scaled EEG
        '''
        eeg_out = deepcopy(eeg)
        # Common average ref
        for sample in range(eeg.shape[0]):
            for time in range(eeg.shape[2]):
                eeg_out[sample, :, time] -= np.mean(eeg_out[sample, :, time])
                eeg_out[sample, :, time] /= eeg_out[sample, :, time].std()
        
        # Normalize
        # for sample in range(eeg.shape[0]):
        #     eeg[sample] /= eeg[sample].std()

        return eeg_out
            

    def scale_source(self, source):
        ''' Scales the sources prior to training the neural network.

        Parameters
        ----------
        source : numpy.ndarray
            A 3D matrix of the source data (samples, dipoles, time_points)
        
        Return
        ------
        source : numpy.ndarray
            Scaled sources
        '''
        source_out = deepcopy(source)
        for sample in range(source.shape[0]):
            for time in range(source.shape[2]):
                source_out[sample, :, time] /= np.max(np.abs(source_out[sample, :, time]))

        return source_out