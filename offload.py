from sys import getsizeof
import numpy as np
class Offload():
    def __init__(self,n_speed,s_speed,msize,X,kernel,hparam,bandwidth_up,bandwidth_down):
        super(Offload, self).__init__()
        self.node_speed=n_speed   #GFLOPS
        self.server_speed=s_speed #GFLOPS
        self.memory_size=msize #mb
        self.X=X
        self.kernel=kernel
        self.hparam=hparam
        self.bandwidth_down=bandwidth_down
        self.bandwidth_up=bandwidth_up
    def outShape(self):
        """
            X-numpy array numpy array (n_H_prev, n_W_prev, n_C_prev)
            k-kernel (f, f, n_C_prev, n_C)
            return-convolution output shape
        """
        p=self.hparam["pad"]
        s=self.hparam["stride"]
        n_H,n_W,n_C=self.X.shape
        f,f,n_cp,n_c=self.kernel.shape
        nh=int((n_H+2*p-f)/s+1)
        nw=int((n_W+2*p-f)/s+1)
        return (nh,nw,n_c)

    def amountOfComputation(self):
        """
            X-numpy array numpy array (n_H_prev, n_W_prev, n_C_prev)
            k-kernel numpy array (f, f, n_C_prev, n_C)
            output-amout of computation base on arithmatic operations
        """
        nh,nw,c=self.outShape()
        f,f,knc,nc=self.kernel.shape

        n=f*f*knc   #length of a column
        num_arth_col=2*n-1 #number of arithmatic operations per column
        return num_arth_col*nh*nw*nc
    
    def amountOfData(self):
        """
            return-data size in bytes for offloading
        """
        return getsizeof(self.X)
    def vecShape(self):
        """
            k-kernel numpy array (f, f, n_C_prev, n_C)
            return - vectorized X shape
        """
        
        f,f,knc,c=self.kernel.shape
        nh,nw,c=self.outShape() 

        return (nh*nw,f*f*knc)


    def checkOffload(self,thershold):
        """ 
            threshold- maximum allowed memory precentage
            output-True when need to offload
        """
        size=self.X.itemsize
        x=self.vecShape()
        #amount of memory 
        mem_amount=(size*x[0]*x[1]+getsizeof(self.kernel))/(1024*1024) #vec inputsize + kernel size (=vec kernel size) mb
        #output shape
        out_shape=self.outShape()
        out_data=size*out_shape[0]*out_shape[1]*out_shape[2] #in bytes
        #amount of arithmatic operations
        ts=(self.amountOfComputation()/self.server_speed)+ (out_data/self.bandwidth_down) + (self.amountOfData()/self.bandwidth_up)
        tn=self.amountOfComputation()/self.node_speed
        if tn>ts:
            return True
        elif mem_amount>self.memory_size*thershold:
            return True
        else:
            print('tn '+str(tn)+' ts '+str(ts))
            return False
        

print('==========offload test===============================================')
np.random.seed(1)
s_speed=5.04*(10**12) #tesla server GFL0PS
n_speed=5.3*(10**9) #raspbery pi 3 GFLOPS
msize=1024 #1GB
#(n_H_prev, n_W_prev, n_C_prev)
X=np.random.randn(64,64,128)
#(f, f, n_C_prev, n_C)
kernel=np.random.randn(9,9,128,256)
hparam={"pad":0,"stride":1}
bandwidth_up=1583113.456 #bytespersec
bandwidth_down=1583113.456
threshold=0.7
print('Node Speed :'+str(n_speed)+"GHz")
print('Server Speed :'+ str(s_speed)+"GHz")
print('Memory Size :'+str(msize)+ "bytes")
print('Input Matix X :'+str(X.shape))
print('Kernel :'+str(kernel.shape))
print('Hparams :'+str(hparam))
print('bandwidth up :'+str(bandwidth_up)+ " bandwidth down :"+str(bandwidth_down))

f=Offload(n_speed,s_speed,msize,X,kernel,hparam,bandwidth_up,bandwidth_down)

print('=============outshape test================================')
outshape=f.outShape()
print('Out shape :' +str(outshape))

print('=============amount of computation test===================')
out2=f.amountOfComputation()
print('Out shape :' +str(out2))

print('=============amount of data===============================')
out3=f.amountOfData()
print('Amount of Data :' +str(out3)+"bytes")

print('===========check X vec shape==============================')
out4=f.vecShape()
print('X vectorized shape: '+str(out4))

print('===========test offload===================================')
print("Threshold value :"+str(threshold))
print('Offload decision :'+str(f.checkOffload(threshold)))
