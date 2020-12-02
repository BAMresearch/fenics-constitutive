import constitutive.cpp
import numpy as np

assert np.all(constitutive.cpp.times_two([1, 2, 3]) == [2, 4, 6])
