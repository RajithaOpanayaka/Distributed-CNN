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

def vecKernel(kernel):
  """
  kernel-- numpy array of shape (n_C, n_C_prev, f, f)
  """
  #size=kernel.itemsize
  #np.lib.stride_tricks.as_strided(x, shape = (5, 4), strides = (8,8))
  n_C,n_C_prev,f,f=kernel.shape
  return kernel.reshape(n_C,f*f*n_C_prev)



def vecConv(X,kernel,hparameters):
    """
    X -- python numpy array of shape (m, n_C, n_H, n_W) representing a batch of m images
    kernel-- numpy array of shape (n_C, n_C_prev, f, f)
    hparameters-- python dictinory containing stride and pad
    """
    pad=hparameters["pad"]
    s=hparameters["stride"]

    X=zero_pad(X, pad)

    out=im2colStride(X[0,:,:,:],kernel[0,:,:,:],s)

    kf=kernel.shape[2] #kernel window size
    n_H= X.shape[2]
    n_W= X.shape[3]
    wh=1+(n_H-kf)//s
    wx=1+(n_W-kf)//s
    n_C=X.shape[1]

    inp=out.reshape(wx*wh,kf*kf*n_C) #vectorized input
    ker=vecKernel(kernel).T  #vectorized kernel
    conv=np.dot(inp,ker)  #vectorize convolution         
    return conv.reshape(kernel.shape[0],wh,wx)
    #return out.reshape(wx*wh,kf*kf*n_C)


def Pooling(X,hparameters,mode="max"):
    """
    X- numpy array (n_C, n_H, n_W)
    hparameters-"f" and "stride"
    """
    strided = np.lib.stride_tricks.as_strided
    n_C,n_H,n_W=X.shape
    f=hparameters["f"]
    s=hparameters["stride"]
    nc,nh,nw=X.strides
    out = strided(X, shape=(n_C,1+(n_H-f)//s,1+(n_W-f)//s,f,f), strides=(nc,nh*s,nw*s,nh,nw))
    vecout = out.reshape(n_C,1+(n_H-f)//s,1+(n_W-f)//s,f*f)
    if mode=="max":
        return np.amax(vecout,axis=3)
    elif mode=="average":
        return np.average(vecout,axis=3)
    

