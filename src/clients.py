from utils import generated_test_distribution
from models import get_net
import torch.optim
import torch.nn as nn

#Client datastructure
class Client():
    def __init__(self, id,net,train_lr,optimizer,criterion):
        self.id = id
        self.net=net
        self.train_loader=train_lr

        classes=[]
        for data in train_lr.dataset:
          if(data[1] not in classes):
            classes.append(data[1])
        self.test_loader=generated_test_distribution(classes, int(len(train_lr.dataset)*0.15)) #specific testset for each clients

        self.optimizer=optimizer
        self.criterion=criterion
        if(len(train_lr.dataset)%100==0):    # an epoch for every 100 images
          self.local_epoch=int(len(train_lr.dataset)/100)
        else:
          self.local_epoch=int(len(train_lr.dataset)/100)+1
        
        self.updates=0    #Δθ memorizes the updates after training
        

#Istance list of clients
def get_clients_list(NUM_CLIENTS,LR,MOMENTUM,train_loader_list):
    clients_list=[]
    for i in range(NUM_CLIENTS):
      net=get_net()
      opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM)
      crt=nn.CrossEntropyLoss()
      client=Client(i,net,train_loader_list[str(i)],opt,crt)
      clients_list.append(client)