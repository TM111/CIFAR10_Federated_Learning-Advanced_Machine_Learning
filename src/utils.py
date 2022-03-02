from torchvision import transforms as transforms
import torchvision
import torch
import copy
import random
import torch.backends.cudnn as cudnn
from models import evaluate

def get_dataset(args):
    train_transform =  transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.BATCH_SIZE, shuffle=False)
    
    # Check dataset sizes
    print('Train Dataset: {}'.format(len(train_set)))
    print('Test Dataset: {}'.format(len(test_set)))
    return test_set, train_set, train_loader, test_loader


#https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/utils.py

#Calculate AVG for each clients layers
def average_weights(updates,clients):           # AggregateClient()
    #IMPLEMENTATION OF FEDAVG
    total_samples=0
    for c in clients:
      total_samples=total_samples+len(c.train_loader.dataset)

    updates_avg = copy.deepcopy(updates[0])
    for key in updates_avg.keys():
        updates_avg[key] = updates_avg[key]*0

    for key in updates_avg.keys():
        for i in range(0, len(updates)):
            updates_avg[key] = updates_avg[key] + updates[i][key]*len(clients[i].train_loader.dataset)
        updates_avg[key] = torch.div(updates_avg[key], total_samples)
    return updates_avg


#CLIENTS -> MAIN MODEL & AVERAGE
def send_client_models_to_server_and_aggregate(main_model,clients,args):
  local_updates=[]
  for c in clients:
    local_updates.append(copy.deepcopy(c.updates))

  updates = average_weights(local_updates,clients) # AggregateClient()

  w=main_model.state_dict()
  for key in w.keys():
    updates[key] = w[key]-args.SERVER_LR*updates[key]    # θt+1 ← θt - γgt
  main_model.load_state_dict(copy.deepcopy(updates))
  return main_model

#PRINTS FOR DEBUG
def print_weights(clients,main_model,args):   # test to view if the algotihm is correct
    w=[]
    for c in clients:
      w.append(c.net.state_dict())
    w_avg=main_model.state_dict()
    if(args.MODEL=='LeNet5'):
      s=''
      for i in range(len(clients)):
        s=s+'size '+str(len(clients[i].train_loader.dataset))+' '+str(w[i]["conv1.weight"][0][0][0][0])+' '+str(clients[i].updates["conv1.weight"][0][0][0][0])+'     '
      print(s)
      print('avg '+str(w_avg["conv1.weight"][0][0][0][0]))
    elif(args.MODEL=='mobilenetV2'):
      s=''
      for i in range(len(clients)):
        s=s+'size '+str(len(clients[i].train_loader.dataset))+' '+str(w[i]["features.2.conv.1.1.weight"][0])+'     '
      print(s)
      print('avg '+str(w_avg["features.2.conv.1.1.weight"][0]))

#MAIN MODEL -> CLIENTS
def send_server_model_to_clients(main_model, clients):
    with torch.no_grad():
      w=main_model.state_dict()
      for i in range(len(clients)):
        clients[i].net.load_state_dict(copy.deepcopy(w))
    return clients



#TRAIN ALL CLIENTS
def train_clients(clients,args):
    for i in range(len(clients)): 
        clients[i].net = clients[i].net.to(args.DEVICE) # this will bring the network to GPU if DEVICE is cuda
        if(i==0):
            previousW=copy.deepcopy(clients[i].net.state_dict())  #save the weights     θ ← θt

        cudnn.benchmark # Calling this optimizes runtime
        for epoch in range(clients[i].local_epoch):    
            for images, labels in clients[i].train_loader:
              # Bring data over the device of choice
              images = images.to(args.DEVICE)
              labels = labels.to(args.DEVICE)

              clients[i].net.train() # Sets module in training mode

              clients[i].optimizer.zero_grad() # Zero-ing the gradients
              # Forward pass to the network
              outputs = clients[i].net(images)
              # Compute loss based on output and ground truth
              loss = clients[i].criterion(outputs, labels)

              # Compute gradients for each layer and update weights
              loss.backward()  # backward pass: computes gradients
              clients[i].optimizer.step() # update weights based on accumulated gradients

        #calculate update   Δθ ← θt - θ
        updates=clients[i].net.state_dict()
        for key in updates.keys():
          updates[key] = previousW[key] - updates[key]
        clients[i].updates=copy.deepcopy(updates)
    return clients


#WEIGHTED ACCURACY
def weighted_accuracy(clients,args):
  sum=0
  num_samples=0
  for i in range(len(clients)):
    test_loss, test_accuracy = evaluate(clients[i].net,clients[i].criterion, clients[i].test_loader,args)
    w=len(clients[i].train_loader.dataset)
    num_samples=num_samples+w

    sum=sum+test_accuracy*w
  return sum/num_samples


#SELECT CLIENTS
def select_clients(clients,args):
  for i in range(random.randint(2,7)):
    random.shuffle(clients)
  round_clients_list=[]
  for i in range(args.NUM_SELECTED_CLIENTS):
    round_clients_list.append(clients[i])
  return round_clients_list