import numpy as np


def softmax(Z):
    """
    Softmax activation function, vectorized version (array Z).
    Args:
        Z (ndarray): numpy array of any shape
    """
    Z_exp = np.exp(Z)
    A = Z_exp / np.sum(Z_exp, axis=1,keepdims=True)

    assert (A.shape == Z.shape)

    return A

def relu(Z):
    """
    ReLU activation function.
    Args:
        Z (ndarray): numpy array of any shape, output of the linear layer
    Returns:
        A (ndarray): post-activation output of relu(Z), same shape as Z
    """
    A = np.maximum(0, Z)
    return A

def ActivationFunc(X,name):
    if name=="relu":
        return relu(X)
    elif name=="softmax":
        return softmax(X)

np.random.seed(1)
x=np.random.randn(2,2,1)
ans=ActivationFunc(x,"softmax")
print(ans)
