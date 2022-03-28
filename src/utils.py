from torchvision import transforms as transforms
import torchvision
import torchvision.datasets
import torch
import copy
import random
import torch.backends.cudnn as cudnn
from models import evaluate
from options import ARGS


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




#https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/utils.py
#Calculate AVG for each clients updates
def average_weights(updates,clients):           # AggregateClient()

    updates_avg = copy.deepcopy(updates[0])
    for key in updates_avg.keys():
        updates_avg[key] = updates_avg[key]*0
    
    if(ARGS.FEDVC):  #standard average
        for key in updates_avg.keys():
            for i in range(0, len(updates)):
                updates_avg[key] = updates_avg[key] + updates[i][key]
            updates_avg[key] = torch.div(updates_avg[key], len(clients))
            
    else:  #weighted average
        total_samples=0
        for c in clients:
          total_samples=total_samples+len(c.train_loader.dataset)
    
        for key in updates_avg.keys():
            for i in range(0, len(updates)):
                updates_avg[key] = updates_avg[key] + updates[i][key]*len(clients[i].train_loader.dataset)
            updates_avg[key] = torch.div(updates_avg[key], total_samples)
            
    return updates_avg




previous_updates=None   #vt-1  for FedAvgM algorithm
#CLIENTS UPDATES -> SERVER & AVERAGE
def send_client_updates_to_server_and_aggregate(server_model,clients):
  local_updates=[]
  for c in clients:
    local_updates.append(copy.deepcopy(c.updates))

  updates = average_weights(local_updates,clients) # AggregateClient()
  
  if(ARGS.ALGORITHM=='FedAvgM'):
      global previous_updates
      if(previous_updates is not None):
          for key in updates.keys():
              updates[key]=ARGS.SERVER_MOMENTUM*previous_updates[key] + updates[key]
      previous_updates=copy.deepcopy(updates)

  w=server_model.state_dict()
  for key in w.keys():
    w[key] = w[key]-ARGS.SERVER_LR*updates[key]    # θt+1 ← θt - γgt
  server_model.load_state_dict(copy.deepcopy(w))
  return server_model





#PRINTS FOR DEBUG
def print_weights(clients,server_model):   # test to view if the algotihm is correct
    w=[]
    for c in clients:
      w.append(c.net.state_dict())
    w_avg=server_model.state_dict()
    
    max_u=-1
    for key in clients[0].updates.keys():
        if('bias' not in key and 'weight' not in key):
            continue
        u=abs(round(torch.sum(clients[0].updates[key]).tolist(),3))
        if u>max_u:
            max_u=u
            node=key
            break

    for i in range(len(clients)):
      s=str(i+1)+')'+'S:'+str(len(clients[i].train_loader.dataset))
      we=str(round(torch.sum(w[i][node]).tolist(),3))
      u=str(round(torch.sum(clients[i].updates[node]).tolist(),3))
      s=s+' W:'+we+' U:'+u+' '
      print(s)
    print('avg '+str(round(torch.sum(w_avg[node]).tolist(),4)))





#SERVER MODEL -> CLIENTS
def send_server_model_to_clients(main_model, clients):
    with torch.no_grad():
      w=main_model.state_dict()
      for i in range(len(clients)):
        clients[i].net.load_state_dict(copy.deepcopy(w))
    return clients



#FEDPROX
def difference_models_norm_2(model_1, model_2): # ||w - w^t||^2
    """Return the norm 2 difference between the two model parameters
    """
    
    tensor_1=list(model_1.parameters())
    tensor_2=list(model_2.parameters())
    
    norm=sum([torch.sum((tensor_1[i]-tensor_2[i])**2) 
        for i in range(len(tensor_1))])
    
    return norm


#TRAIN ALL CLIENTS
def train_clients(clients):
    for i in range(len(clients)): 
        #SET MODEL
        clients[i].net = clients[i].net.to(ARGS.DEVICE) # this will bring the network to GPU if DEVICE is cuda
        clients[i].net.train()
        cudnn.benchmark # Calling this optimizes runtime
        if i == 0:
            previous_model=copy.deepcopy(clients[0].net)  #save the weights     θ ← θt
            previousW=previous_model.state_dict()  #save the weights     θ ← θt
        
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
              
              if ARGS.MU>0: #FEDPROX https://epione.gitlabpages.inria.fr/flhd/federated_learning/FedAvg_FedProx_MNIST_iid_and_noniid.html
                  loss = loss + ARGS.MU/2*difference_models_norm_2(clients[i].net,previous_model) 
              
              # Compute gradients for each layer and update weights
              loss.backward()  # backward pass: computes gradients
              clients[i].optimizer.step() # update weights based on accumulated gradients

        #CALCULATE UPDATE   Δθ ← θt - θ
        updates=clients[i].net.state_dict()
        for key in updates.keys():
          updates[key] = previousW[key] - updates[key]
        clients[i].updates=copy.deepcopy(updates)
        
    return clients






#WEIGHTED ACCURACY
def weighted_accuracy(clients):
  sum=0
  num_samples=0
  for i in range(len(clients)):
    loss, accuracy = evaluate(clients[i].net, clients[i].test_loader)
    sum=sum+accuracy*len(clients[i].train_loader.dataset)
    num_samples=num_samples+len(clients[i].train_loader.dataset)
  return sum/num_samples


#SELECT CLIENTS
def select_clients(clients):
    
  if(ARGS.FEDVC==True): #select clients with probabilities
      indexes_range=list(range(len(clients)))

      W=[]
      for i in indexes_range:
          W.append(len(clients[i].train_loader.dataset))
          
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