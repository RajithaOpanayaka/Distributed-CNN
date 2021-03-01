import pickle
import socket
import json
import time
import numpy as np
import struct
from weights import kernels
from vec import vecConv,Pooling
from activation import ActivationFunc

s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)

print('FOG Socket Created')

s.bind(('localhost',9999))

s.listen(1) #number of connections 1
payload_size = struct.calcsize("L")  ### CHANGED
data = b''

def send(c,data):
    data_string=pickle.dumps(data)
    message_size = struct.pack("L", len(data_string))
    c.sendall(message_size+data_string)


def receive_array(data,payload_size,conn):
        while len(data) < payload_size:
            data += conn.recv(4096)

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]

        # Retrieve all data based on message size
        while len(data) < msg_size:
            data += conn.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Extract frame
        frame = pickle.loads(frame_data)
        return frame


while True:
    """
        data_variable={"data":X,"pos":a,"layer":layer}
        a=(start,end)
        layer={"l_type":"conv","kernel":"W1","hparams":{"stride":1,"pad":0}}
        pooling layer testing
        layer2={"l_type":"max","hparams":{"stride":1,"f":2}}

        conv_dict={ "data":X,"index":index}

    """
    #CNN architecture goes here
    CNN=[{"l_type":"conv","kernel":"W1","bias":"b1","hparams":{"stride":1,"pad":2},"act":"relu"},{"l_type":"max","hparams":{"stride":1,"f":2}}]

    c,addr =s.accept()
    #receive data from client
    tic = time.process_time()
    data_variable=receive_array(data,payload_size,c)
    print('Connect with',addr)

    X=data_variable["data"]
    index=data_variable["index"]
    
    #FOG node computation goes here
    for index,layer in enumerate(CNN[index:],index):
        mode=layer["l_type"]
        hparam=layer["hparams"]
        #check offload
        offload=False
        if(offload):
            #offload to the cloud
            #X=
            break
        else:
            if(mode=='conv'):
                w=kernels["kernel"]
                X=vecConv(X,w,hparam)
                X+=kernels["bias"]
                X=ActivationFunc(X,layer["act"])
            else:
                X=Pooling(X,hparam,mode)

    #X - output send to roof node
    dout={"data":X}
    send(c,dout)
    toc = time.process_time()
    print ("Computation time FOG server = " + str(1000*(toc - tic)) + "ms")
    c.close()
