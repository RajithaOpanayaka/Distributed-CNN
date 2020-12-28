from sys import getsizeof
class Offload():
    bandwidth_up=0
    bandwidth_down=0
    def __init__(self,n_speed,s_speed,msize,X,kernel,hparam):
        super(Offload, self).__init__()
        self.node_speed=n_speed
        self.server_speed=s_speed
        self.memory_size=msize
        self.X=X
        self.kernel=kernel
        self.hparam=hparam
    def outShape(self):
        """
            X-numpy array numpy array (m,n_C, n_H, n_W)
            k-kernel numpy array (f,f,n_C,c)
            return-convolution output shape
        """
        p=self.hparam["pad"]
        s=self.hparam["stride"]
        m,n_C,n_H,n_W=self.X.shape
        f,f,knc,c=self.kernel.shape
        nh=((n_H+2*p-f)/s)+1
        nw=((n_W+2*p-f)/s)+1
        return (c,nh,nw)

    def amountOfComputation(self):
        """
            X-numpy array numpy array (m,n_C, n_H, n_W)
            k-kernel numpy array (f,f,n_C,c)
            output-amout of computation base on arithmatic operations
        """
        p=self.hparam["pad"]
        s=self.hparam["stride"]
        m,n_C,n_H,n_W=self.X.shape
        f,f,knc,c=self.kernel.shape
        nh=((n_H+2*p-f)/s)+1
        nw=((n_W+2*p-f)/s)+1

        n=f*f*n_C   #length of a column
        num_arth_col=2*n-1 #number of arithmatic operations per column
        return num_arth_col*nh*nw*c
    
    def amountOfData(self):
        """
            return-data size in bytes for offloading
        """
        return getsizeof(self.X)
    def vecShape(self):
        """
            X-numpy array (m,n_C, n_H, n_W)
            return - vectorized X shape
        """
        p=self.hparam["pad"]
        s=self.hparam["stride"]
        m,n_C,n_H,n_W=self.X.shape
        f,f,knc,c=self.kernel.shape
        nh=((n_H+2*p-f)//s)+1
        nw=((n_W+2*p-f)//s)+1

        n=f*f*n_C   #length of a column

        #return {x:(nh*nw,f*f*n_C),k:(c,f*f*n_C)}
        return (nh*nw,f*f*n_C)


    def checkOffload(self,thershold):
        """ 
            threshold- maximum allowed memory precentage
            output-True when need to offload
        """
        n_CPI=1
        s_CPI=1
        size=self.X.itemsize
        x=vecShape()
        #amount of memory 
        mem_amount=size*x[0]*x[1]+getsizeof(self.kernel) #vec inputsize + kernel size (=vec kernel size)
        #output shape
        out_shape=outShape()
        out_data=size*out_shape[0]*out_shape[1]*out_shape[2]
        #amount of arithmatic operations
        ts=(amountOfComputation()*n_CPI/self.server_speed)+ (out_data/self.bandwidth_down) + (amountOfData()/self.bandwidth_up)
        tn=amountOfComputation()*n_CPI/self.node_speed
        if tn>ts:
            return True
        elif mem_amount>self.memory_size*thershold:
            return True
        else:
            return False
        