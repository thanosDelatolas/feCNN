import mne
from copy import deepcopy
from . import simulation

from . import util
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
        self.subject = sources.subject if type(sources) == mne.SourceEstimate \
            else sources[0].subject