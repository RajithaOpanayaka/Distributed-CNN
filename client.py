import socket

c = socket.socket()

c.connect(('localhost',6001)) #ip address and port
c.send("From clinet")
print(c.recv(1024).decode())