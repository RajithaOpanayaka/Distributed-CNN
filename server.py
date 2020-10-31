import socket

s=socket.socket()

print('Socket Created')

s.bind(('localhost',6001))

s.listen(1) #number of connections 1

while True:
	c,addr =s.accept()
	data=c.recv(1024)
	print('Connect with',addr,data)
	c.send("Welcome to server")

	c.close()



