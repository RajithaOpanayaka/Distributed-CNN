from client import client
from offload import Offload
import numpy as np
from vec import vecConv,Pooling
from weights import kernels
from activation import ActivationFunc
"""
input CNN
 
"""
class Master:
    """
        list of layers of CNN
        list of IP address of server nodes
        edge server ip address for offloading
    """
    def __init__(self,CNN,nodes,edge,image):
        super(Master, self).__init__()
        self.CNN=CNN
        self.nodes=nodes
        self.edge=edge
        self.image=image
        self.count=0
    def thread_Compute(self,X,layer):
        """
            X-numpy array(n_H_prev, n_W_prev, n_C_prev)
            Distribute and return output 
        """
        threads=[]
        #========get the corresponding weights length============
        #pooling n_c=input matirx channels
        #conv n_c =the weights channels
        if layer["l_type"]=="conv":
            n_C=kernels[layer["kernel"]].shape[3]
        else:
            n_C=X.shape[2]
        
        #======================
        pos=self.getPos(n_C,0,len(self.nodes))
        start,end=pos
        for node in self.nodes:
            start,end=self.getPos(n_C,end,len(self.nodes))
            a=(start,end)
            if layer["l_type"]=="conv":
                conv_dict = {"data":X,"pos":a,"layer":layer}
            else:
                conv_dict = {"data":X[:,:,a[0]:a[1]],"layer":layer}
            threads.append(client(conv_dict,node["ip"],node["port"]))
        for t in threads:
            t.start()
        out= self.layerResult(layer,X,pos)
        for t in threads:
            t.join()
            out=np.concatenate((out,t.value()["data"]), axis=2)
        self.count=0

        return out
    
    def getPos(self,len,end,nodes):
        """
            len-length of array
            end-last partition point
            count-assinged nodes
            nodes-number of server nodes
        """
        start=end
        if self.count==nodes:
            last=len
        else:
            last=end+(int)(len/(nodes+1))
        self.count+=1
        return (start,last)
    
    def layerResult(self,layer,X,pos):
        """ 
            layer-dictonary l_type,kernel,hparams
            X-numpy array(n_H_prev, n_W_prev, n_C_prev)
            pooling-
                X-numpy array(n_H_prev, n_W_prev, n_C_prev)
                hparameters-"f" and "stride"
                Pooling(X,hparameters,mode="max")
            conv
                vecConv(X,kernel,hparameters):
                X- numpy arrya shape (n_H_prev, n_W_prev, n_C_prev)
                kernel-numpy array of shape (f, f, n_C_prev, n_C)
                hparameters-- python dictinory containing stride and pad
        """
        if(layer["l_type"]=="conv"):
            w=kernels[layer["kernel"]]
            hparam=layer["hparams"]
            return vecConv(X,w[:,:,:,pos[0]:pos[1]],hparam)
        else:
            hparam=layer["hparams"]
            mode=layer["l_type"]
            #batch size of 1
            return Pooling(X[:,:,pos[0]:pos[1]],hparam,mode)
    def poolingMem(self,X,hparam):
        n_H,n_W,n_C=X.shape
        f=hparam["f"]
        s=hparam["stride"]
        return X.itemsize*(f*f)*(1+(n_H-f)//s)*(1+(n_W-f)//s)*n_C
    
    def compute(self,n_speed,s_speed,msize,thershold,bandwidth):
        """
            n_speed-node CPU speed
            s_speed-server CPU speed
            msize-node memory size
            thershold-maximum memory limit
        """
        X=self.image
        for index,layer in enumerate(self.CNN):
            #layer dictonary {type,kernel,bias,hparams}
            #offloading decisions
            kernel=kernels[layer["kernel"]]
            hparam=layer["hparams"]
            if(layer["l_type"]=="conv"):
                #n_speed,s_speed,msize,X,kernel,hparam,bandwidth_up,bandwidth_down
                off_dec=Offload(n_speed,s_speed,msize,X,kernel,hparam,bandwidth,bandwidth)
                offDec=off_dec.checkOffload(thershold)
            else:
                #pooling layer memory usage
                offDec=self.poolingMem(X,hparam)
            if X.shape[2]<50:
                pos=0
                if layer[l_type]=="conv":
                    pos=(0,kernel.shape[3])
                else:
                    pos=(0,X.shape[2])
                X=self.layerResult(layer,X,pos)
            elif (offDec):
                #get the result form the server
                conv_dict={ "data":X,"index":index}
                c=client(conv_dict,self.edge["ip"],self.edge["port"])
                c.send()
                X=c.receive_array()
            else:
                X=self.thread_Compute(X,layer)
            
            #activation function and bais apply for conv
            if layer[l_type]=="conv":
                X+=kernels[layer["bias"]]
                X= ActivationFunc(X,layer["act"])
        
        #YOLO layers


            
                

            

#########################################################
CNN=[{"l_type":"conv","kernel":"W1","bias":"b1","hparams":{"stride":1,"pad":2},"act":"relu"},{"l_type":"max","hparams":{"stride":1,"f":2}}]
nodes=[{"ip":"localhost","port":9999}]
edge={"ip":"localhost","port":9000}
image=np.array([1,2])
master_node=Master(CNN,nodes,edge,image)
#master_node.compute(2.3,3.3,1000,50)
# """
# W_test=np.ones((2,2,1,1))
# b1_test=np.zeros((1,1,1,1))
# hparameters1_test={"pad" : 0,"stride": 1}
# """
# X=np.ones((1,3,3,1))
# #conv testing
# layer={"l_type":"conv","kernel":"W1","hparams":{"stride":1,"pad":0}}
# #pooling layer testing
# layer2={"l_type":"max","hparams":{"stride":1,"f":2}}
# print('=======getPos=========================')
# print(master_node.getPos(1,0,1))
# pos=master_node.getPos(kernels[layer["kernel"]].shape[3],0,0)
# print(pos)
# print('=====layer result=====================')
# result=master_node.layerResult(layer,X,pos)
# result2=master_node.layerResult(layer2,X,pos)
# print(result)
# print(result.shape)
# print('=======pooling layer=================')
# print(result2)
# print(result2.shape)

print('============thread test===============')
#thread_Compute(self,X,layer)
#W_test=np.ones((2,2,1,1))
np.random.seed(1)
X1= np.random.randn(3, 3, 1)
layer1={"l_type":"conv","kernel":"W1","hparams":{"stride":1,"pad":0}}
result3=master_node.thread_Compute(X1,layer1)
print(result3.shape)
print(result3)
#master_node.thread_Compute(X,layer)


# print('============vecConv test==================')
# out=vecConv(X1[0,:,:,:],kernels["W1"],{"stride":1,"pad":0})
# print(out.shape)
# print(out)