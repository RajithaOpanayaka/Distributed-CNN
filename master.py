from client import client
from offload import offload

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
            off_dec=offload(n_speed,s_speed,msize,X,kernel,hparam)
            if(checkOffload(thershold)):
                #get the result form the server
                pass
                
            else:
                for node in self.nodes:
                    c=client(layer)
                    c.start()   
                c.join()
                #X=np.concatenate((out,c.value()["data"]), axis=3)

            

