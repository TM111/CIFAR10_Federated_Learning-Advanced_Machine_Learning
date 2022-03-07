from sampling import generated_test_distribution
from models import get_net
import torch.optim
import torch.nn as nn
from options import ARGS
from utils import get_dataset_distribution

#Client datastructure
class Client():
    def __init__(self, id,net,train_lr,optimizer,criterion,local_epochs,test_lr):
        self.id = id
        self.net=net
        self.train_loader=train_lr
        self.test_loader=test_lr #specific testset for each clients

        self.optimizer=optimizer
        self.criterion=criterion
        self.local_epochs=local_epochs
        
        self.updates=None    #Δθ memorizes the updates after last training

#Istance list of clients
def get_clients_list(train_loader_list, train_set, test_set):
    clients_list=[]
    
    if(ARGS.FEDIR==True):
        p=get_dataset_distribution(train_set)
        
    for i in range(ARGS.NUM_CLIENTS):
      
      if(len(train_loader_list[str(i)].dataset)%100==0):    # an epoch for every 100 images
        local_epochs=int(len(train_loader_list[str(i)].dataset)/100)
      else:
        local_epochs=int(len(train_loader_list[str(i)].dataset)/100)+1
        
      classes=[]
      for data in train_loader_list[str(i)].dataset:
        if(data[1] not in classes):
          classes.append(data[1])
      num_samples=int(len(train_loader_list[str(i)].dataset)*0.2)
      test_loader=generated_test_distribution(classes, test_set,num_samples) #specific testset for each clients
        
      net=get_net()
      opt = torch.optim.SGD(net.parameters(), lr=ARGS.LR, momentum=ARGS.MOMENTUM)
      w=None
      
      if(ARGS.FEDIR==True):
          q=get_dataset_distribution(train_loader_list[str(i)].dataset)
          w=[0 for label in range(ARGS.NUM_CLASSES)]
          for label in range(ARGS.NUM_CLASSES):
              if(q[label]==0):  #se il client non ha quella classe???
                  w[label]=0
              else:
                  w[label]=p[label]/q[label]
          w=torch.tensor(w)

      crt=nn.CrossEntropyLoss(weight=w)
      
      
      client=Client(i,net,train_loader_list[str(i)],opt,crt,local_epochs,test_loader)
      clients_list.append(client)
    return clients_list