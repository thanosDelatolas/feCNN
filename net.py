from abc import abstractmethod
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Conv1D) 
from copy import deepcopy
from sklearn.model_selection import train_test_split
import datetime
import visualkeras

import pickle
import numpy as np

import losses

time_str = datetime.datetime.now().strftime('%d_%m_%Y__%H:%M:%S')

class NN:
    ''' The neural network class that creates and trains the  model. 
    
        Attributes
        ----------
        sim : The simulation object located in simulation.py

        eeg_topographies : ndarray (n_samples,67,67)
            A set of topographies for each eeg signal in simulation. This argument is necessary for the CNN only.

        activation_function : str or tf.keras.activations
            The activation function used for each fully connected layer.

        Methods
        -------
        fit : trains the neural network with the EEG and source data
        train : trains the neural network with the EEG and source data
        predict : perform prediciton on EEG data
        evaluate : evaluate the performance of the model
    '''

    def __init__(self, sim, activation_function='ReLU', verbose=True):
        self.sim = deepcopy(sim)

        self.n_elec = self.sim.fwd.leadfield.shape[0]
        self.n_dipoles = self.sim.fwd.leadfield.shape[1]

        # simulation's samples
        self.n_samples = self.sim.eeg_data.shape[1]
        self.activation_function = activation_function
        self.compiled = False

        self.default_loss = losses.weighted_huber_loss

        self.verbose = verbose
        self.trained = False

    @abstractmethod
    def build_model(self):
        ''' Build the neural network architecture.       
        '''
        pass

    @abstractmethod
    def fit(self, learning_rate=0.01, 
        validation_split=0.2, epochs=50, 
        false_positive_penalty=2, delta=1., batch_size=100, 
        loss=None, patience=5
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
            false_positive_penalty : float
                Defines weighting of false-positive predictions. Increase for conservative 
                inverse solutions, decrease for liberal prediction.
            batch_size : int
                The number of samples to simultaneously calculate the error 
                during backpropagation.
            loss : tf.keras.losses
                The loss function.
            patience: int
                Number of epochs with no improvement after which trainning will be stopped
            Return
            ------
            history, tensorboard dir : keras.callbacks.History, string
                Method returns the history and the log dir for the tensorboard.

        '''
        pass
    
    def predict(self, eeg):
        ''' Predict the sources from the eeg.
        '''
        if self.trained:
            predicted_sources = self.model.predict(eeg)
            return predicted_sources
        
        else:
            raise AttributeError('The model must be trained first.')

    @abstractmethod
    def evaluate_nmse(self, eeg, sources):
        ''' Evaluate the model regarding normalized mean squared error
        
        Parameters
        ----------         
            eeg : numpy.ndarray
                The simulated EEG data
            sources : numpy.ndarray
                The simulated EEG data

        Return
        ------
        normalized_mean_squared_errors : numpy.ndarray
            The normalized mean squared error of each sample
        '''

        pass

    @abstractmethod
    def evaluate_mse(self, eeg, sources):
        ''' Evaluate the model regarding mean squared error
        
        Parameters
        ----------         
            eeg : numpy.ndarray
                The simulated EEG data
            sources : numpy.ndarray
                The simulated EEG data

        Return
        ------
            mean_squared_errors : numpy.ndarray
            The mean squared error of each sample
        '''
        
        pass



    def save_nn(self, model_filename, save_sim=False, sim_filename='sim.pkl'):
        if self.trained:
            self.model.save(model_filename)

            if save_sim:
                with open(sim_filename, 'wb') as outp:  # Overwrites any existing file.
                    pickle.dump(self.sim, outp, pickle.HIGHEST_PROTOCOL)

        else:
            print('The model must be trained first.')
    
    def load_nn(self, model_filename):
        ''' This function loads the neural network.

            !Important!
            -----------
            The simulation object must be the same with the one that it was used during trainning. 
        '''
        self.model = load_model(model_filename,  compile=False)
        self.compiled = True
        self.trained = True
        print('Loaded model in', self.__class__.__name__,':')
        self.model.summary()

    @staticmethod
    def lr_schedule(epoch):
        '''
            Returns a custom learning rate that decreases as epochs progress.
        '''
        learning_rate = 0.075
        if epoch > 10:
            learning_rate = 0.02
        if epoch > 20:
            learning_rate = 0.01
        if epoch > 40:
            learning_rate = 0.005

        tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
        return learning_rate

    
class EEGMLP(NN):
    '''  An MLP nueral network
   
    '''
    
    def build_model(self):
        ''' Build the neural network architecture using the 
        tensorflow.keras.Sequential() API. 

        The architecture is an MLP with three hidden layers.
       
        '''
        if not self.compiled :
            # Build the artificial neural network model using Dense layers.
            self.model = keras.Sequential()

            # add input layer
            self.model.add(keras.Input(shape=(self.n_elec,), name='InputLayer'))

            # first hidden layer with 256 neurons.
            self.model.add(Dense(units=256, activation=self.activation_function, name='Hidden-1'))

            # second hidden layer with 512 neurons.
            self.model.add(Dense(units=512, activation=self.activation_function, name='Hidden-2'))

            # third hidden layer with 1024 neurons.
            self.model.add(Dense(units=1024, activation=self.activation_function, name='Hidden-3'))            

            # Add output layer
            self.model.add(Dense(self.n_dipoles, activation='linear', name='OutputLayer'))

            if self.verbose:
                self.model.summary()
                img = './assets/MLP.png'
                img_keras = './assets/MLP-visual-keras.png'
                tf.keras.utils.plot_model(self.model, to_file=img, show_shapes=True)
                visualkeras.layered_view(self.model, legend=True,  to_file=img_keras)  

    def fit(self, learning_rate=0.01, 
        validation_split=0.2, epochs=50, 
        false_positive_penalty=2, delta=1., batch_size=100, 
        loss=None, patience=5
    ):

        if len(self.sim.eeg_data.shape) != 2 :
            raise AttributeError("EEG data must be 2D (n_elctrodes x n_samples")
        elif len(self.sim.source_data.shape) != 2 :
            raise AttributeError("Sources data must be 2D (n_dipoles x n_samples")

        tensorboard_dir = 'logs/MLP-Model-{}'.format(time_str)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)
        
        # Input data
        x = self.sim.eeg_data.T        

        # Target data
        y = self.sim.source_data.T        

        # early stoping
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', \
            mode='min', patience=patience, restore_best_weights=True, verbose=1)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


        if loss == None:
            loss = self.default_loss(weight=false_positive_penalty, delta=delta)
        
        metrics = [tf.keras.metrics.MeanAbsoluteError(name="MAE"), 
            tf.keras.metrics.RootMeanSquaredError(name="RMSE"),
            #tf.keras.metrics.MeanAbsolutePercentageError(name="MAPE")            
        ]

        if not self.compiled:
            self.model.compile(optimizer, loss, metrics=metrics)
            self.compiled = True
        
        data_train, data_val, labels_train, labels_val = train_test_split(x, y, test_size=validation_split, shuffle=True)

        del x, y

        history = self.model.fit(data_train, labels_train, 
                epochs=epochs, batch_size=batch_size, shuffle=False, 
                validation_data=(data_val, labels_val), callbacks=[es, tensorboard_callback])
        self.trained = True
        
        return history, tensorboard_dir

    
    def evaluate_nmse(self, eeg, sources):
        ''' Evaluate the model regarding normalized mean squared error
        '''

        if self.trained:
            if eeg.shape[1]  != sources.shape[1]:
                raise AttributeError('EEG and Sources data must have the same amount of samples.')
            predicted_sources = self.predict(eeg=eeg.T).T

            sources /= np.max(sources)
            predicted_sources /= np.max(predicted_sources)
            
            normalized_mean_squared_errors = np.mean((predicted_sources - sources)**2, axis=0)
            return normalized_mean_squared_errors
            
        else :
            print('The model must be trained')
    
    def evaluate_mse(self, eeg, sources):
        ''' Evaluate the model regarding mean squared error

        '''
        
        if self.trained:

            if eeg.shape[1]  != sources.shape[1]:
                raise AttributeError('EEG and Sources data must have the same amount of samples.')

            predicted_sources = self.predict(eeg=eeg.T).T        
            mean_squared_errors = np.mean((predicted_sources - sources)**2, axis=0)

            return mean_squared_errors
        else :
            print('The model must be trained first.')


class EEG_CNN(NN):
    ''' A CNN to solve the inverse problem.

        The additional eeg_topographies argument is a set of topographies for each eeg signal
        in simulations.
    '''

    def __init__(self, sim, eeg_topographies ,activation_function='ReLU', verbose=True):
        super().__init__(sim, activation_function, verbose)

        if len(eeg_topographies.shape) != 3 :
            raise AttributeError('The set of topographies must be a 3D-array. (n_samples x xpxiels x ypixels)')
        elif eeg_topographies.shape[0] != self.n_samples :
            raise AttributeError('Incompatible sim and topographies.')
        
        self.eeg_topographies = deepcopy(eeg_topographies)


    def build_model(self):
        ''' Build the neural network architecture using the 
        tensorflow.keras.Sequential() API. 

        The architecture is a CNN with three hidden layers.
       
        '''
        if not self.compiled :
            # Build the artificial neural network model using Dense layers.
            self.model = keras.Sequential()

            # add input layer
            self.model.add(keras.Input(shape=(self.eeg_topographies.shape[1], self.eeg_topographies.shape[2],1), name='Input'))
            self.model.add(Conv2D(8, kernel_size=(3, 3), activation='sigmoid'))
            # self.model.add(Dropout(0.25))
            self.model.add(Flatten())
            self.model.add(Dense(512, activation='sigmoid'))
            self.model.add(Dense(1024, activation='sigmoid'))
            self.model.add(Dense(2048, activation='sigmoid'))
            # Add output layer
            self.model.add(Dense(self.n_dipoles, activation='linear', name='OutputLayer'))

            if self.verbose:
                self.model.summary()
                img = './assets/CNN.png'
                img_keras = './assets/CNN-visual-keras.png'
                tf.keras.utils.plot_model(self.model, to_file=img, show_shapes=True)
                visualkeras.layered_view(self.model, legend=True,  to_file=img_keras)  
    
    def fit(self, learning_rate=0.01, 
        validation_split=0.2, epochs=50,
        false_positive_penalty=2, delta=1., batch_size=100, 
        loss=None, patience=5
    ):

        if len(self.sim.eeg_data.shape) != 2 :
            raise AttributeError("EEG data must be 2D (n_elctrodes x n_samples")
        elif len(self.sim.source_data.shape) != 2 :
            raise AttributeError("Sources data must be 2D (n_dipoles x n_samples")

        tensorboard_dir = 'logs/CNN-Model-{}'.format(time_str)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)

        # Input data
        x = self.eeg_topographies       

        # Target data
        y = self.sim.source_data.T        

        # early stoping
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', \
            mode='min', patience=patience, restore_best_weights=True,verbose=1)

        #optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        optimizer = tf.keras.optimizers.SGD()

        if loss == None:
            # loss = self.default_loss(weight=false_positive_penalty, delta=delta)
            loss = 'MSE'

        metrics = [tf.keras.metrics.MeanAbsoluteError(name="MAE"), 
            tf.keras.metrics.RootMeanSquaredError(name="RMSE"),
            #tf.keras.metrics.MeanAbsolutePercentageError(name="MAPE")            
        ]

        lr_callback = keras.callbacks.LearningRateScheduler(NN.lr_schedule)

        if not self.compiled:
            self.model.compile(optimizer, loss, metrics=metrics)
            self.compiled = True

        history = self.model.fit(x, y, 
                epochs=epochs, batch_size=batch_size, shuffle=True, 
                validation_split=validation_split, callbacks=[es, tensorboard_callback, lr_callback])
        self.trained = True
        
        return history, tensorboard_dir

    
    def evaluate_nmse(self, eeg, sources):
        ''' Evaluate the model regarding normalized mean squared error
        
            The eeg parapameter must be the eeg- topographies.
        '''

        if self.trained:

            if len(eeg.shape) != 3 :
                raise AttributeError('The set of topographies must be a 3D-array. (n_samples x xpxiels x ypixels)')
            elif eeg.shape[0] != sources.shape[1] :
                raise AttributeError('Incompatible sim and topographies.')

            predicted_sources = self.predict(eeg=eeg).T

            sources /= np.max(sources)
            predicted_sources /= np.max(predicted_sources)
            
            normalized_mean_squared_errors = np.mean((predicted_sources - sources)**2, axis=0)
            return normalized_mean_squared_errors
            
        else :
            print('The model must be trained')

    
    def evaluate_mse(self, eeg, sources):
        ''' Evaluate the model regarding mean squared error
        
            The eeg parameter must be the eeg-topographies
        '''
        
        if self.trained:

            if len(eeg.shape) != 3 :
                raise AttributeError('The set of topographies must be a 3D-array. (n_samples x xpxiels x ypixels)')
            elif eeg.shape[0] != sources.shape[1] :
                raise AttributeError('Incompatible sim and topographies.')
           

            predicted_sources = self.predict(eeg=eeg).T        
            mean_squared_errors = np.mean((predicted_sources - sources)**2, axis=0)

            return mean_squared_errors
        else :
            print('The model must be trained first.')