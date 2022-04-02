from models import get_net_and_optimizer
from options import ARGS
import copy

#Server datastructure
class Server():
    def __init__(self):
        self.model, _ = get_net_and_optimizer()
        self.previous_updates=None   #vt-1  for FedAvgM algorithm
        if(ARGS.ALGORITHM=='SCAFFOLD'):
            self.c_global=copy.deepcopy(self.model.state_dict()) # server control variates (SCAFFOLD)
            for key in self.c_global: 
                self.c_global[key] = self.c_global[key]*0.0

