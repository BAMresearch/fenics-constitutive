import numpy as np
import cofe

a1 = np.arange(9, dtype=np.float64)

test = cofe.Test(a1, 42.)
print("there")
test.do()
