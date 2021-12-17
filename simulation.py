from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm
import numpy as np
import random
import colorednoise as cn

import util


# Specifications about the sources.
DEFAULT_SETTINGS = {
    'number_of_sources': 1 , # (1,5) , int/tuple/list
    'extents': (21, 80), # int/float/tuple/list,   diameter of sources (in mm)
    'amplitudes': (5, 10), #  int/float/tuple/list, the electrical current of the source in nAm
    'shapes': 'gaussian', # str,  How the amplitudes evolve over space. Can be 'gaussian' for now.
}

class Simulation:
    ''' Simulate EEG and sources data.
    '''

    def __init__(self, fwd, settings=DEFAULT_SETTINGS, source_data=None, eeg_data=None,parallel=False, n_jobs=-1):
        self.settings = settings
        self.check_settings()

        self.fwd = fwd
        self.parallel = parallel
        self.n_jobs = n_jobs

        # if source_data and eeg_data are already known from previous simulations
        self.source_data = source_data
        self.eeg_data = eeg_data

        if self.source_data is not None and self.eeg_data is not None and \
            self.source_data.shape[1] == self.eeg_data.shape[1]:
                self.simulated = True
        else :
            self.simulated = False
            self.source_data = None
            self.eeg_data = None
    
    
    def create_train_dataset(self, times_each_dipole):
        ''' This method creates the trainning dataset which has 50460 * times_each_dipole  source spaces. 
            Each dipole is selected times_each_dipole as a seed location while the electrical current of the dipole 
            and the extent of the activation are selected randomly.

            The shape of the simulated sources will be times_each_dipole * 50460, 50460.

            Hence, times_each_dipole * 50460 different source spaces will be created.
        '''
        if self.simulated:
            print('The data are already simulated.')
            return

        n_dipoles = self.fwd.leadfield.shape[1]
        n_samples = times_each_dipole * n_dipoles
        print('Sources Simulation')
        sources = np.stack([self.simulate_source(src_center=dipole % n_dipoles) \
            for dipole in tqdm(range(n_samples))], axis=0)
        
        self.source_data = sources
        # The eeg data will be created from matlab
        #self.eeg_data = self.simulate_eeg()
        
        self.simulated = True
        

    def simulate(self, n_samples=10000):
        ''' Simulate sources and EEG data'''
        if self.simulated :
            print('The data are already simulated.')
            return

        self.source_data = self.simulate_sources(n_samples)
        self.eeg_data = self.simulate_eeg()        
        self.simulated = True


    def simulate_sources(self, n_samples):
        
        if self.simulated :
            print('The data are already simulated.')

            return self.source_data
        print('Simulate Sources.')
        if self.parallel:
            source_data = np.stack(Parallel(n_jobs=self.n_jobs, backend='loky')
                (delayed(self.simulate_source)() for _ in range(n_samples)))
        else:
            source_data = np.stack([self.simulate_source() for _ in tqdm(range(n_samples))], axis=0)

        source_data = source_data.T
        return source_data

    def simulate_source(self, src_center=-1):
        ''' Returns a vector containing the dipole currents. Requires only a 
        dipole position list and the simulation settings.

        Parameters (located in the settings dict).
        ----------
        forward : The forward object located in forward.py
        number_of_sources : int/tuple/list
            number of sources. Can be a single number or a list of two 
            numbers specifying a range.
        extents : int/float/tuple/list
            diameter of sources (in mm). Can be a single number or a list of 
            two numbers specifying a range.
        amplitudes : int/float/tuple/list
            the electrical current of the source in nAm
        shapes : str
            How the amplitudes evolve over space. Can be 'gaussian' for now.
        src_center : int, 
            if src_center is already decided. If not -1 then it will be chosen randomly.
        Return
        ------
        source : numpy.ndarray, (n_dipoles,), the simulated value of its dipole
        '''

        # Get a random sources number in range:
        number_of_sources = self.get_from_range(self.settings['number_of_sources'], dtype=int)

        # Get the diameter for each source
        extents = [self.get_from_range(self.settings['extents'], dtype=float) for _ in range(number_of_sources)]

        # Decide shape of sources
        if self.settings['shapes'] == 'gaussian':
            shapes = [self.settings['shapes']] * number_of_sources
        else :
            raise AttributeError('Only Gaussian shape is supported!')


        # Get amplitude gain for each source (amplitudes come in nAm)
        amplitudes = [self.get_from_range(self.settings['amplitudes'], dtype=float) * 1e-9 for _ in range(number_of_sources)]

        # Get source centers
        if src_center != -1 :
            src_centers = np.random.choice(np.arange(self.fwd.leadfield.shape[1]), \
                number_of_sources, replace=False)
        else :
            src_centers = [src_center]

        signal_length = 1
        signals = [np.array([1])]*number_of_sources
        
        source = np.zeros((self.fwd.leadfield.shape[1], signal_length))

        ##############################################
        # Loop through source centers (i.e. seeds of source positions)
        for i, (src_center, shape, amplitude, signal) in enumerate(zip(src_centers, shapes, amplitudes, signals)):
            dists = np.sqrt(np.sum((self.fwd.dipoles - self.fwd.dipoles[src_center, :])**2, axis=1))
            d = np.where(dists<extents[i]/2)[0]

            if shape == 'gaussian':
                sd = np.clip(np.max(dists[d]) / 2, a_min=0.1, a_max=np.inf)  # <- works better
                activity = np.expand_dims(util.gaussian(dists, 0, sd) * amplitude, axis=1) * signal
                source += activity
            else:
                msg = BaseException("shape must be of type >string< and be >gaussian<")
                raise(msg)

        return np.squeeze(source)

    def simulate_eeg(self):
        ''' Create EEG of specified number of trials based on sources and some SNR.
        Parameters
        -----------
        fwd : Forward
            the Forward object located in forward.py
        n_jobs : int
                Number of jobs to run in parallel. -1 will utilize all cores.

        Return
        -------
        simulated eeg data : numpy.ndarray
            3D array of shape (n_elec x n_samples x timepoints)
        '''
        if self.simulated :
            print('The data are already simulated.')
            return
        
        print('Simulate EEG.')

        # Desired Dim of sources: (samples x dipoles x time points)
        sources = self.source_data

        # calculate eeg 
        eeg_clean = np.array(self.project_sources(sources))


        # eeg_noisy = self.add_noise_to_eeg(eeg_clean)            
       
        return eeg_clean

    def project_sources(self, sources):
        ''' Project sources through the leadfield to obtain the EEG data.
        Parameters
        ----------
        sources : numpy.ndarray
            3D array of shape (n_dipoles x n_samples x timepoints)
        
        Return the eeg signlas
        ------
        '''
        print('Project sources to EEG.')
        leadfield = self.fwd.leadfield

        # Scale to allow for lower precision
        # scaler = 1/sources_tmp.max()
        # sources_tmp *= scaler

        # Perform Matmul
        result = np.matmul(leadfield.astype(np.float32), sources.astype(np.float32))

        # Rescale
        # result /= scaler

        return result

    def add_noise_to_eeg(eeg):
        ''' This function adds noise to the eeg signal
        '''

        print('Add noise to EEG ...')

    def check_settings(self):
        ''' Check if settings are complete and insert missing 
            entries if there are any.
        '''
        # Check for wrong keys:
        for key in self.settings.keys():
            if not key in DEFAULT_SETTINGS.keys():
                msg = f'key {key} is not part of allowed settings. See DEFAULT_SETTINGS for reference: {DEFAULT_SETTINGS}'
                raise AttributeError(msg)
        
        # Check for missing keys and replace them from the DEFAULT_SETTINGS
        for key in DEFAULT_SETTINGS.keys():
            # Check if setting exists and is not None
            if not (key in self.settings.keys() and self.settings[key] is not None):
                self.settings[key] = DEFAULT_SETTINGS[key]


    @staticmethod
    def get_from_range(val, dtype=int):
        ''' If list of two integers/floats is given this method outputs a value in between the two values.
        Otherwise, it returns the value.
        
        Parameters
        ----------
        val : list/tuple/int/float

        Return
        ------
        out : int/float

        '''
        if dtype==int:
            rng = random.randrange
        elif dtype==float:
            rng = random.uniform
        else:
            msg = f'dtype must be int or float, got {type(dtype)} instead'
            raise AttributeError(msg)

        if isinstance(val, (list, tuple, np.ndarray)):
            out = rng(*val)
        elif isinstance(val, (int, float)):
            out = val
        return out


    @staticmethod
    def get_pulse(pulse_len):
        ''' Returns a pulse of given length. A pulse is defined as 
        half a revolution of a sine.
        
        Parameters
        ----------
        x : int
            the number of data points

        '''
        pulse_len = int(pulse_len)
        freq = (1/pulse_len) / 2
        time = np.arange(pulse_len)

        signal = np.sin(2*np.pi*freq*time)
        return signal
    
    