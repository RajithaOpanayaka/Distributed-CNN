import numpy as np
from skimage.util.shape import view_as_windows
def zero_pad(X, pad):
    """
    X -- python numpy array of shape (n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (n_H + 2*pad, n_W + 2*pad, n_C)
    """
    
    ### START CODE HERE ### (≈ 1 line)
    X_pad =np.pad(X, ((pad,pad),(pad,pad),(0,0)), mode='constant', constant_values = (0,0))
    ### END CODE HERE ###
    
    return X_pad

def im2colStride(X,kernel,s):
    """
    X-numpy array (n_H, n_W, n_C)
    kernel- numpy array (f, f, n_C_prev, n_C)
    s- stride

    """
    return view_as_windows(X, kernel.shape, step=s)

def vecKernel(kernel):
  """
  kernel-numpy array (f, f, n_C_prev, n_C)
  """
  
  f,f,n_C_prev,n_C=kernel.shape
  return kernel.reshape(f*f*n_C_prev,n_C)

def vecConv(X,kernel,hparameters):
    """
    X- numpy arrya shape (n_H_prev, n_W_prev, n_C_prev)
    kernel-numpy array of shape (f, f, n_C_prev, n_C)
    hparameters-- python dictinory containing stride and pad
    """
    pad=hparameters["pad"]
    s=hparameters["stride"]

    X=zero_pad(X, pad)

    out=im2colStride(X[:,:,:],kernel[:,:,:,0],s)

    kf=kernel.shape[0] #kernel window size
    n_H= X.shape[0]
    n_W= X.shape[1]
    wh=1+(n_H-kf)//s
    wx=1+(n_W-kf)//s
    n_C=X.shape[2]

    inp=out.reshape(wx*wh,kf*kf*n_C) #vectorized input
    ker=vecKernel(kernel)  #vectorized kernel
    conv=np.dot(inp,ker)  #vectorize convolution         
    return conv.reshape(wh,wx,kernel.shape[3])

def Pooling(X,hparameters,mode="max"):
    """
    X-numpy array(n_H_prev, n_W_prev, n_C_prev)
    hparameters-"f" and "stride"
    """
    strided = np.lib.stride_tricks.as_strided
    n_H,n_W,n_C=X.shape
    f=hparameters["f"]
    s=hparameters["stride"]
    nh,nw,nc=X.strides
    out = strided(X, shape=(f,f,1+(n_H-f)//s,1+(n_W-f)//s,n_C), strides=(nh,nw,nh*s,nw*s,nc))
    vecout = out.reshape(f*f,1+(n_H-f)//s,1+(n_W-f)//s,n_C)
 
    if mode=="max":
        return np.amax(vecout,axis=0)
    elif mode=="average":
        return np.average(vecout,axis=0)