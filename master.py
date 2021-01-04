from client import client
from offload import offload
import numpy as np
from vec import vecConv,Pooling
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
    def thread_Compute(X,layer):
        """
            Distribute and return output 
        """
        threads=[]
        for node in self.nodes:
            a=1
            conv_dict = {"data":X,"pos":a,"layer":layer}
            threads.append(client(conv_dict,node["ip"],node["port"]))
        for t in threads:
            t.start()
        out= layerResult(layer,X,pos)
        for t in threads:
            t.join()
            out=np.concatenate((out,t.value()["data"]), axis=3)

        return out
    
    def layerResult(layer,X,pos):
        if(layer["l_type"]=="conv"):
            w=layer["kernel"]
            hparam=layer["hparams"]
            return vecConv(X,w[:,:,:,:pos],hparam)
        else:
            hparams=layer["hparams"]
            mode=layer["l_type"]
            return Pooling(X,hparams,mode)
    
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
                    conv_dict={ "data":X,"l_type":layer[l_type],"hpara":,"pos":0}
                    c=client(conv_dict,self.edge["ip"],self.edge["port"])
                    c.send()
                    X=c.receive_array()
                    
                else:
                    X=thread_Compute(X,layer)

            else:
                X=thread_Compute(X,layer)
                

            

#########################################################
CNN=[{"l_type":"conv","kernel":"W1","hparams":{stride:1,pad:2}},{"l_type":"Max_pool","hparams":{stride:1,f:2}}]
nodes=[{"ip":"192.168.1.1","port":9999}]
edge={"ip":"192.168.1.1","port":9999}
image=np.array([1,2])
master_node=Master(CNN,nodes,edge,image)
master_node.compute(2.3,3.3,1000,50)