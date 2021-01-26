import numpy as np
np.random.seed(1)
image=np.random.randn(1, 256, 256, 3) #h256X256 image
W1=np.random.randn(3, 3, 3, 32)
b1=np.random.randn(1, 1, 1, 32)
W2=np.random.randn(3, 3, 32, 64)
b2=np.random.randn(1, 1, 1, 64)
W3=np.random.randn(3, 3, 64, 128)
b3=np.random.randn(1, 1, 1, 128)
hparameters1 = {"pad" : 129,"stride": 2}
hparameters3 = {"pad" : 65,"stride": 2}
hparameters2 = {"stride" : 2, "f": 2}
hparameters4 = {"stride" : 2, "f": 2}
hparameters5 = {"pad" : 32,"stride": 2}

W_test=np.ones((1,3,3,1))
b1_test=np.random.randn(1,1,1,1)
hparameters1_test={"pad" : 0,"stride": 1}
kernels={"W1":W_test,"b1":b1_test,"hparam":hparameters1_test}

