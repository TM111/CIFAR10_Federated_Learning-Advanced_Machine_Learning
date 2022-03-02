from sampling import generated_test_distribution
from models import get_net
import torch.optim
import torch.nn as nn

#Client datastructure
class Client():
    def __init__(self, id,net,train_lr,optimizer,criterion,test_lr):
        self.id = id
        self.net=net
        self.train_loader=train_lr
        self.test_loader=test_lr #specific testset for each clients

        self.optimizer=optimizer
        self.criterion=criterion
        if(len(train_lr.dataset)%100==0):    # an epoch for every 100 images
          self.local_epoch=int(len(train_lr.dataset)/100)
        else:
          self.local_epoch=int(len(train_lr.dataset)/100)+1
        
        self.updates=0    #Δθ memorizes the updates after training
        

#Istance list of clients
def get_clients_list(train_loader_list,test_set, args):
    clients_list=[]
    for i in range(args.NUM_CLIENTS):
      net=get_net(args)
      opt = torch.optim.SGD(net.parameters(), lr=args.LR, momentum=args.MOMENTUM)
      crt=nn.CrossEntropyLoss()
      
      classes=[]
      for data in train_loader_list[str(i)].dataset:
        if(data[1] not in classes):
          classes.append(data[1])
      num_samples=int(len(train_loader_list[str(i)].dataset)*0.2)
      test_loader=generated_test_distribution(classes, test_set,num_samples,args) #specific testset for each clients
      
      client=Client(i,net,train_loader_list[str(i)],opt,crt,test_loader)
      clients_list.append(client)
    return clients_list