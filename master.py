from client import client
from offload import offload
import numpy as np
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
		super(client, self).__init__()
		self.CNN=CNN
        self.nodes=nodes
        self.edge=edge
        self.image=image
    
    def compute(n_speed,s_speed,msize,thershold):
        """
            n_speed-node CPU speed
            s_speed-server CPU speed
            msize-node memory size
            thershold-maximum memory limit
        """
        X=self.image
        for layer in self.CNN:
            #layer dictonary {type,kernel,bias,hparams}
            #offloading decisions
            kernel=layer["kernel"]
            hparam=layer["hparams"]
            if layer[l_type]=="conv":
                off_dec=offload(n_speed,s_speed,msize,X,kernel,hparam)
            if(off_dec.checkOffload(thershold)):
                #get the result form the server
                pass   
            else:
                for node in self.nodes:
                    c=client(layer)
                    c.start()   
                c.join()
                #X=np.concatenate((out,c.value()["data"]), axis=3)

            

#########################################################
CNN=[{l_type:"conv",kernle:"W1",hparams:{stride:1,pad:2}},{l_type:"Max_pool",hparams:{stride:1,f:2}}]
nodes=["192.168.1.1"]
edge=["192.168.1.1"]
image=np.array([1,2])
master_node=Master(CNN,nodes,edge,image)
master_node.compute(2.3,3.3,1000,50)