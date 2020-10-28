#YOLO conv net
import time
from yolo import conv_forward,pool_forward
from weights import image,W1,b1,W2,b2,W3,b3,hparameters1,hparameters2,hparameters3,hparameters4,hparameters5
tic = time.process_time()
out1=conv_forward(image, W1, b1, hparameters1) #3x3 s-1 pad-1 filters 16 activation-leaky
out2=pool_forward(out1, hparameters2, mode = "max") #2x2 s-2
out3=conv_forward(out2, W2, b2, hparameters3) #3x3 s-1 pad-1 filters 32 activation-leaky
out4=pool_forward(out3, hparameters4, mode = "max") #2x2 s-2
out5=conv_forward(out4, W3, b3, hparameters5) #3x3 s-1 pad-1 filters 32 activation-leaky
toc = time.process_time()
print ("Computation time = " + str(1000*(toc - tic)) + "ms")
out5.shape
