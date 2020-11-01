import pickle
import socket

s=socket.socket()

print('Socket Created')

s.bind(('localhost',6001))

s.listen(1) #number of connections 1

while True:
	c,addr =s.accept()
	data=c.recv(4096)
	data_variable=pickle.loads(data)
	print('Connect with',addr,data_variable.data.shape)
	c.send("Welcome to server")

	c.close()



