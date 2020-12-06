import numpy as np
from skimage.util.shape import view_as_windows

def zero_pad(X, pad):
    """
    Argument:
    X -- numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_C,n_H + 2*pad, n_W + 2*pad)
    """
    X_pad =np.pad(X, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant', constant_values = (0,0))

    return X_pad

def im2colStride(X,kernel,s):
    """
    X -- numpy array of shape (n_C,n_H, n_W) 
    kernel-- numpy array of shape (n_C_prev,f, f)
    s-- stride

    """
    return view_as_windows(X, kernel.shape, step=s)

def vecConv(X,kernel,hparameters):
    """
    X -- python numpy array of shape (m, n_C, n_H, n_W) representing a batch of m images
    kernel-- numpy array of shape (n_C, n_C_prev, f, f)
    hparameters-- python dictinory containing stride and pad
    """
    pad=hparameters["pad"]
    s=hparameters["stride"]

    zero_pad(X, pad)

    out=im2colStride(X[0,:,:,:],kernel[0,:,:,:],s)

    kf=kernel.shape[2] #kernel window size
    n_H= X.shape[2]
    n_W= X.shape[3]
    wh=1+(n_H-kf)//s
    wx=1+(n_W-kf)//s
    n_C=X.shape[1]

    return out.reshape(wx*wh,kf*kf*n_C)
