import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np

duneuropy_out = os.path.join(parent,'duneuropy/DataOut')

leadfield = np.load(os.path.join(duneuropy_out, 'solution.npy'))

print(leadfield.shape)