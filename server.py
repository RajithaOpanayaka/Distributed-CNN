import pickle
import socket
import json
import numpy as np
import struct
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
    c,addr =s.accept()
    #receive data from client
    data_variable=receive_array(data,payload_size,c)
    print('Connect with',addr,data_variable["data"].shape)
    send(c,data_variable)
    #send data to client
	#c.send(bytes("Welcome to server",'utf-8'))
    c.close()



