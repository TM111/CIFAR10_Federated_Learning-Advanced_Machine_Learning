from torchvision import transforms as transforms
import torchvision
import torchvision.datasets
import torch
import copy
import random
import torch.backends.cudnn as cudnn
from models import evaluate,get_net_and_optimizer
from options import ARGS
import torch.nn as nn


def get_dataset():
    train_transform =  transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=ARGS.BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=ARGS.BATCH_SIZE, shuffle=True)
    
    # Check dataset sizes
    print('Train Dataset: {}'.format(len(train_set)))
    print('Test Dataset: {}'.format(len(test_set)))
    return train_loader.dataset, test_loader.dataset, train_loader, test_loader




#Calculate AVG for each clients updates
def average_weights(Server, n_list, local_updates_list, tau_list, c_delta_list):           # AggregateClient()

    total_n=sum(n for n in n_list)
    
    updates_avg = copy.deepcopy(local_updates_list[0])
    for key in updates_avg.keys():
        updates_avg[key] = updates_avg[key]*0.0

    
    if(ARGS.ALGORITHM == 'FedNova'): #https://github.com/Xtra-Computing/NIID-Bench
        a_i_list=[]
        rho=ARGS.MOMENTUM
        for tau in tau_list:
            a_i = (tau - rho * (1 - rho**tau) / (1 - rho)) / (1 - rho)
            a_i_list.append(a_i)
        
        for i in range(len(local_updates_list)):
            for key in local_updates_list[i].keys():
                updates_avg[key] = updates_avg[key] + local_updates_list[i][key] * n_list[i] / a_i_list[i]
        for key in updates_avg.keys():
            updates_avg[key] = updates_avg[key] / total_n 
        
        coeff = 0.0   # τeff
        for i in range(len(n_list)):
            coeff = coeff + a_i_list[i] * n_list[i] / total_n
        ARGS.SERVER_LR=coeff
        
    elif(ARGS.ALGORITHM in ['FedAvg','FedAvgM','FedSGD','FedProx','SCAFFOLD']):
        if(ARGS.FEDVC):  #standard average
            for key in updates_avg.keys():
                for i in range(len(local_updates_list)):
                    updates_avg[key] = updates_avg[key] + local_updates_list[i][key]
                updates_avg[key] = torch.div(updates_avg[key], len(local_updates_list))
                
        else:  #weighted average
            for key in updates_avg.keys():
                for i in range(len(local_updates_list)):
                    updates_avg[key] = updates_avg[key] + local_updates_list[i][key]*n_list[i]
                updates_avg[key] = torch.div(updates_avg[key], total_n)

        if(ARGS.ALGORITHM == 'SCAFFOLD'): #https://github.com/Xtra-Computing/NIID-Bench
            # Update c_global
            Server.model = Server.model.to('cpu')
            
            total_delta = copy.deepcopy(Server.model.state_dict())
            for key in total_delta:
                total_delta[key] = total_delta[key]*0.0
                
            for c_delta in c_delta_list:
                for key in total_delta:
                    total_delta[key] = total_delta[key] + c_delta[key]
            
            for key in Server.c_global:
                Server.c_global[key] = Server.c_global[key] + total_delta[key] / len(c_delta_list)
            
    return updates_avg




#CLIENTS UPDATES -> SERVER & AVERAGE
def send_client_updates_to_server_and_aggregate(Server,clients):
    # Information to be sent to server
    n_list=[]                   # size of local dataset
    local_updates_list=[]
    tau_list=[]                 # tau for FedNova (optimizer step)
    c_delta_list=[]             # c_deltas for SCAFFOLD
    for client in clients:
        n_list.append(len(client.train_loader.dataset))
        local_updates_list.append(client.updates)
        if(ARGS.ALGORITHM=='FedNova'):
            tau_list.append(client.tau)
        elif(ARGS.ALGORITHM=='SCAFFOLD'):
            c_delta_list.append(client.c_delta)
    # Aggregate
    updates = average_weights(Server, n_list, local_updates_list, tau_list, c_delta_list) 
    # Update global model
    Server.model = Server.model.to('cpu')
    for name, w in Server.model.named_parameters():
        w.grad=updates[name]
    Server.optimizer.step()
    return Server



#PRINTS FOR DEBUG
def print_weights(clients,server_model):   # test to view if the algotihm is correct
    w=[]
    for c in clients:
      w.append(c.net.state_dict())
    w_avg=server_model.state_dict()
    print("")
    for key in clients[0].updates.keys():
        if('bias' in key or 'weight' in key):
            node=key
            break
    total_s=0.0
    numeratore=0.0
    for i in range(len(clients)):
        
      s=str(i+1)+')'+'S:'+str(len(clients[i].train_loader.dataset))
      we=str(round(torch.sum(w[i][node]).tolist(),3))
      u=str(round(torch.sum(clients[i].updates[node]).tolist(),3))
      s=s+' W:'+we+' U:'+u+' '
      print(s)
      
      total_s+=len(clients[i].train_loader.dataset)
      numeratore+=torch.sum(w[i][node]).tolist()*len(clients[i].train_loader.dataset)
    print('avg '+str(round(torch.sum(w_avg[node]).tolist(),4)))
    print('cor',round(numeratore/total_s,4))


#SERVER MODEL -> CLIENTS
def send_server_model_to_clients(Server, clients):
    w=Server.model.state_dict()
    for i in range(len(clients)):
      clients[i].net.load_state_dict(copy.deepcopy(w)) # send global model to clients
      if(ARGS.ALGORITHM=='SCAFFOLD'): # send c_global to clients
          clients[i].c_global=Server.c_global
    return clients



#TRAIN ALL CLIENTS
def train_clients(clients):
    for i in range(len(clients)): 
        if (i==0):
              clients[i].net = clients[i].net.cpu()
              previous_model=copy.deepcopy(clients[0].net)  #save global model
              
        if(ARGS.ALGORITHM=='FedProx'):
              previous_model = previous_model.to(ARGS.DEVICE)
        
        if(ARGS.ALGORITHM=='SCAFFOLD'):
            for key in clients[i].c_global:
                  clients[i].c_global[key] = clients[i].c_global[key].to(ARGS.DEVICE)
                  clients[i].c_local[key] = clients[i].c_local[key].to(ARGS.DEVICE)
              
        #SET MODEL
        clients[i].net.to(ARGS.DEVICE) # this will bring the network to GPU if DEVICE is cuda
        clients[i].net.train()
        cudnn.benchmark # Calling this optimizes runtime
        

        #SET LOADER AND EPOCH
        if(ARGS.FEDVC):
            if(len(clients[i].train_loader.dataset)>=ARGS.NVC):
                indexes=random.sample(range(len(clients[i].train_loader.dataset)), ARGS.NVC)
            else:
                indexes=[random.randint(0,len(clients[i].train_loader.dataset)-1) for idx in range(ARGS.NVC)]
            loader = torch.utils.data.DataLoader(dataset=torch.utils.data.Subset(clients[i].train_loader.dataset,indexes), batch_size=ARGS.BATCH_SIZE)
            epochs = clients[i].local_epochs
            
        else:
            loader=clients[i].train_loader
            epochs=clients[i].local_epochs

        #START TRAIN
        for epoch in range(epochs):    
            for images, labels in loader:
              # Bring data over the device of choice
              images = images.to(ARGS.DEVICE)
              labels = labels.to(ARGS.DEVICE)
              clients[i].net.train() # Sets module in training mode

              clients[i].optimizer.zero_grad() # Zero-ing the gradients
              # Forward pass to the network
              outputs = clients[i].net(images)
              # Compute loss based on output and ground truth
              loss = clients[i].criterion(outputs, labels)

              if(ARGS.ALGORITHM=='FedProx'): # https://epione.gitlabpages.inria.fr/flhd/federated_learning/FedAvg_FedProx_MNIST_iid_and_noniid.html
                  tensor_1=list(clients[i].net.parameters())
                  tensor_2=list(previous_model.parameters())
                  norm=sum([torch.sum((tensor_1[i]-tensor_2[i])**2) for i in range(len(tensor_1))]) # ||w - w^t||^2
                  loss = loss + ARGS.MU/2*norm
                  
              # Compute gradients for each layer and update weights

              loss.backward()  # backward pass: computes gradients
              clients[i].optimizer.step() # update weights based on accumulated gradients

              if(ARGS.ALGORITHM=='SCAFFOLD'):
                  net_para = clients[i].net.state_dict()
                  for key in net_para:
                      net_para[key] = net_para[key] - ARGS.LR * (clients[i].c_global[key] - clients[i].c_local[key]) # c_global - c_local (variance reduction)
                  clients[i].net.load_state_dict(net_para)

        clients[i].net.cpu()
        previous_model.cpu()
        #CALCULATE UPDATE   Δθ ← θt - θ
        previousW=previous_model.state_dict()  #save the weights     θ ← θt
        updates=clients[i].net.state_dict()
        for key in updates.keys():
          updates[key] = previousW[key] - updates[key]
        clients[i].updates=copy.deepcopy(updates)

        if(ARGS.ALGORITHM=='SCAFFOLD'):
            for key in clients[i].c_global:
                  clients[i].c_global[key] = clients[i].c_global[key].cpu()
                  clients[i].c_local[key] = clients[i].c_local[key].cpu()
            # Update c_local and calculate delta
            c_new = copy.deepcopy(clients[i].c_local)
            for key in clients[i].c_local:
                c_new[key] = c_new[key] - clients[i].c_global[key] + updates[key] / (clients[i].tau * ARGS.LR)
                clients[i].c_delta[key] = c_new[key] - clients[i].c_local[key]
            clients[i].c_local = c_new
    return clients




#WEIGHTED ACCURACY
def weighted_accuracy(clients):
  if(ARGS.CENTRALIZED_MODE):
      return 0
  sum=0
  num_samples=0
  for i in range(len(clients)):
    loss, accuracy = evaluate(clients[i].net, clients[i].test_loader)
    sum=sum+accuracy*len(clients[i].train_loader.dataset)
    num_samples=num_samples+len(clients[i].train_loader.dataset)
  return sum/num_samples


#SELECT CLIENTS
def select_clients(clients):
  if(ARGS.FEDVC): #select clients with probabilities
      indexes_range=list(range(len(clients)))
      W=[len(clients[i].train_loader.dataset) for i in indexes_range]
      indexes=[]
      for i in range(ARGS.NUM_SELECTED_CLIENTS):
          idx=random.choices(indexes_range, weights=W)[0]
          indexes.append(idx)
          W[idx]=0
  else:
      indexes=random.sample(range(0,len(clients)), ARGS.NUM_SELECTED_CLIENTS)
      
  return [clients[i] for i in indexes]


#GET DATASET LABELS DISTRIBUTION
def get_dataset_distribution(dataset):
    dataset_ = torch.utils.data.Subset(dataset,[j for j in range(len(dataset))])
    d=[0 for j in range(ARGS.NUM_CLASSES)]
    for label in range(len(dataset_)):
        index=dataset[label][1]
        d[index]=d[index]+1
    return [round(d[j]/len(dataset),3) for j in range(ARGS.NUM_CLASSES)]
