from sampling import generated_test_distribution
from models import get_net_and_optimizer
import torch.optim
import torch.nn as nn
from options import ARGS
from utils import get_dataset_distribution
import os
import pickle as pickle

#Client datastructure
class Client():
    def __init__(self, id,net,train_lr,test_lr,optimizer,criterion,local_epochs):
        self.id = id
        self.net=net
        self.train_loader=train_lr
        self.test_loader=test_lr #specific testset for each clients

        self.optimizer=optimizer
        self.criterion=criterion
        self.local_epochs=local_epochs
        
        self.updates=net.state_dict() #Δθ memorizes the updates after last training
        for key in self.updates.keys():
          self.updates[key] = self.updates[key]*0
        

#Istance list of clients
def get_clients_list(train_loader_list, train_set, test_set):
    clients_list=[]
    
    if(ARGS.FEDIR):
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
      
      client=Client(i,net,train_loader_list[str(i)],test_loader,opt,crt,ARGS.NUM_EPOCHS)
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