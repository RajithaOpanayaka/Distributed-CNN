#YOLO conv net
# import time
# from yolo import conv_forward,pool_forward
# from weights import image,W1,b1,W2,b2,W3,b3,hparameters1,hparameters2,hparameters3,hparameters4,hparameters5
# tic = time.process_time()
# out1=conv_forward(image, W1, b1, hparameters1) #3x3 s-1 pad-1 filters 16 activation-leaky
# out2=pool_forward(out1, hparameters2, mode = "max") #2x2 s-2
# out3=conv_forward(out2, W2, b2, hparameters3) #3x3 s-1 pad-1 filters 32 activation-leaky
# out4=pool_forward(out3, hparameters4, mode = "max") #2x2 s-2
# out5=conv_forward(out4, W3, b3, hparameters5) #3x3 s-1 pad-1 filters 32 activation-leaky
# toc = time.process_time()
# print ("Computation time = " + str(1000*(toc - tic)) + "ms")
# out5.shape

from offload import Offload
import pytest
import numpy as np
from vec import Pooling

def test_outshape():
    """
    X-numpy array numpy array (m,n_C, n_H, n_W)
    k-kernel (n_C, n_C_prev, f, f)
    out-(m,c,nh,nw)    
    """
    X = np.random.randn(10,4,5,7)
    W = np.random.randn(8,4,3,3)
    hparam = {"pad" : 1,
               "stride": 2}

    obj=Offload(1,1,1,X,W,hparam)
    assert obj.outShape() == (10,8,3,4)



def test_amountOfComputation():
    X = np.random.randn(1,1,4,4)
    W = np.random.randn(1,1,2,2)
    hparam = {"pad" : 0,
               "stride": 1}

    obj=Offload(1,1,1,X,W,hparam)
    assert obj.amountOfComputation()==9*7

def test_vecShape():
    X = np.random.randn(1,2,4,4)
    W = np.random.randn(1,2,2,2)
    hparam = {"pad" : 0,
               "stride": 1}

    obj=Offload(1,1,1,X,W,hparam)
    assert obj.vecShape()==(9,8)


# def test_Pooling():
#     np.random.seed(1)
#     A_prev = np.random.randn(1, 5, 5, 3)
#     hparameters = {"stride" : 1, "f": 2}
#     max_output=[[[ 1.62434536,  0.86540763, -0.52817175],
#          [ 1.74481176,  0.90159072,  0.50249434],
#          [ 1.74481176,  1.46210794,  0.50249434],
#          [ 0.90085595,  1.46210794,  1.13376944]],

#         [[ 0.04221375,  0.58281521, -0.0126646 ],
#          [ 1.14472371,  0.90159072,  1.65980218],
#          [ 1.14472371,  0.90159072,  1.65980218],
#          [ 0.90085595,  1.6924546 ,  0.53035547]],

#         [[ 0.12015895,  0.61720311,  2.10025514],
#          [ 0.12015895,  0.61720311,  1.65980218],
#          [ 0.74204416,  0.58662319,  1.65980218],
#          [ 0.93110208,  1.6924546 ,  0.88514116]],

#         [[ 0.12015895,  1.25286816,  2.10025514],
#          [ 1.13162939,  1.51981682,  2.18557541],
#          [ 1.13162939,  1.51981682,  2.18557541],
#          [ 0.93110208,  0.87616892,  0.88514116]]]
#     assert_array_equal(Pooling(A_prev[0,:,:,:],hparameters,mode="max"),max_output)


