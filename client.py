import pickle
import socket
import numpy as np
np.random.seed(1)
image=np.random.randn(1, 256, 256, 3) #h256X256 image
conv_dict = { 'data':image}
c = socket.socket()

c.connect(('localhost',6001)) #ip address and port
data_string=pickle.dumps(conv_dict)
c.send(data_string)
print(c.recv(1024).decode())