import pickle
import socket
import json
import time
import numpy as np
import struct
import yolo as y
from weights import kernels
from vec import vecConv,Pooling

s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)

print('Socket Created')

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
    """
    c,addr =s.accept()
    #receive data from client
    tic = time.process_time()
    data_variable=receive_array(data,payload_size,c)
    print('Connect with',addr,data_variable["data"].shape)
    #imgout=y.conv_forward(data_variable["data"], w.W1[:,:,:,data_variable["pos"]:], w.b1[:,:,:,data_variable["pos"]:],data_variable["hpara"])
    #out={"data":imgout}
    X=data_variable["data"]
    hparam=data_variable["layer"]["hparams"]
    mode=data_variable["layer"]["l_type"]
    if(mode=="conv"):
        pos=data_variable["pos"]
        w=kernels[data_variable["layer"]["kernel"]]
        out=vecConv(X,w[:,:,:,pos[0]:pos[1]],hparam)
    else:
        out=Pooling(X,hparam,mode)
    dout={"data":out}
    send(c,dout)
    toc = time.process_time()
    print ("Computation time for conv part2 = " + str(1000*(toc - tic)) + "ms")
    #send data to client
	#c.send(bytes("Welcome to server",'utf-8'))
    c.close()
