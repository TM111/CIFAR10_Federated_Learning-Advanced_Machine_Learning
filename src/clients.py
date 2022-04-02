from sampling import generated_test_distribution
from models import get_net_and_optimizer
import torch.optim
import torch.nn as nn
from options import ARGS
from utils import get_dataset_distribution
import os
import pickle as pickle
import copy

#Client datastructure
class Client():
    def __init__(self, id,net,train_lr,test_lr,optimizer,criterion,local_epochs, c_local):
        self.id = id
        self.net=net
        self.train_loader=train_lr
        self.test_loader=test_lr #specific testset for each clients

        self.optimizer=optimizer
        self.criterion=criterion
        self.local_epochs=local_epochs
        
        self.updates=None #Δθ memorizes the updates after last training
        
        self.tau=sum(1 for epoch in range(local_epochs) for batch in train_lr)  # number of optimizer step

        self.c_local=copy.deepcopy(c_local)  # local control variates (SCAFFOLD)
        self.c_delta=copy.deepcopy(c_local)  # delta c (SCAFFOLD)
        self.c_global=copy.deepcopy(c_local) # server control variates (SCAFFOLD)
        

        

#Istance list of clients
def get_clients_list(train_loader_list, train_set, test_set):
    clients_list=[]
    
    if(ARGS.ALGORITHM in ['FedAvg','FedAvgM','FedSGD'] and ARGS.FEDIR):
        p=get_dataset_distribution(train_set)
        
    for i in range(ARGS.NUM_CLIENTS):
      classes=[]
      for data in train_loader_list[str(i)].dataset:
        if(data[1] not in classes):
          classes.append(data[1])
      num_samples=int(len(train_loader_list[str(i)].dataset)*0.2)
      test_loader=generated_test_distribution(classes, test_set,num_samples) #specific testset for each clients
        
      net, opt = get_net_and_optimizer()
      w=None
      
      if(ARGS.ALGORITHM in ['FedAvg','FedAvgM','FedSGD'] and ARGS.FEDIR):
          q=get_dataset_distribution(train_loader_list[str(i)].dataset)
          w=[0 for label in range(ARGS.NUM_CLASSES)]
          for label in range(ARGS.NUM_CLASSES):
              if(q[label]==0):  #se il client non ha quella classe???
                  w[label]=0
              else:
                  w[label]=p[label]/q[label]
          w=torch.tensor(w).to(ARGS.DEVICE)

      crt=nn.CrossEntropyLoss(weight=w)
      
      c_local=None
      if(ARGS.ALGORITHM=='SCAFFOLD'):
          c_local=copy.deepcopy(net.state_dict()) # local control variates (SCAFFOLD)
          for key in c_local: c_local[key] = c_local[key]*0.0
          
      client=Client(i,net,train_loader_list[str(i)],test_loader,opt,crt,ARGS.NUM_EPOCHS, c_local)
      clients_list.append(client)
    
    #cache clients list
    if(ARGS.COLAB):
        dir="/content/clients_list_cache/"
    else:
        dir=os.getcwd()+"/clients_list_cache/"
    file=str(ARGS.DISTRIBUTION)+str(ARGS.ALPHA)+str(ARGS.NUM_CLASS_RANGE[0])+str(ARGS.NUM_CLASS_RANGE[1])+str(ARGS.NUM_CLIENTS)+str(ARGS.RATIO)+str(ARGS.Z)+str(ARGS.FEDIR)
    with open(dir+file, 'wb') as config_dictionary_file:
        pickle.dump(clients_list,config_dictionary_file)
    return clients_list