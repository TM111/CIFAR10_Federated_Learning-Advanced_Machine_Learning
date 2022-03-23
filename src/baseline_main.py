from sampling import dirichlet_distribution, cifar_iid, cifar_noniid, get_train_distribution,get_cached_clients
from utils import get_dataset,send_server_model_to_clients, select_clients, train_clients, send_client_updates_to_server_and_aggregate, print_weights, weighted_accuracy,get_dataset_distribution
from options import ARGS
from clients import get_clients_list
from models import get_net_and_optimizer, evaluate
import torch.optim
import torch.nn as nn
import time
import random


if __name__ == '__main__':
    if(ARGS.COLAB==0): #test locally
        ARGS.MODEL='LeNet5'
        ARGS.DEVICE='cpu'
        ARGS.ALGORITHM='FedAvg'  #FedAvg FedAvgM
        ARGS.FEDIR=False
        ARGS.FEDVC=False
        ARGS.BATCH_NORM=1
        #ARGS.OPTIMIZER='adam'
        ARGS.SERVER_MOMENTUM=1
        ARGS.PRETRAIN=False
        ARGS.FREEZE=False
        ARGS.DISTRIBUTION=4
        ARGS.ALPHA=0.1
        ARGS.RATIO=0.6
        ARGS.Z=4
        #ARGS.CENTRALIZED_MODE=True
        ARGS.NUM_CLIENTS=100
        ARGS.NUM_SELECTED_CLIENTS=3
        ARGS.NUM_EPOCHS=2
        
    if(ARGS.CENTRALIZED_MODE):
        ARGS.DISTRIBUTION=2
        ARGS.NUM_CLIENTS=1
        ARGS.NUM_SELECTED_CLIENTS=1
        ARGS.FEDIR=False
        ARGS.FEDVC=False
    
    
    train_set, test_set, train_loader, test_loader = get_dataset()  #download dataset
    
    clients=get_cached_clients() #get cached clients list
    if(clients==None):
        train_loader_list=get_train_distribution(train_set) #split trainset to current distribution
        clients=get_clients_list(train_loader_list, train_set, test_set) #generate client list
    
    for i in range(random.randint(2,7)):
      random.shuffle(clients)
    ind=0
    print("Distribution of trainset for client "+str(ind),get_dataset_distribution(clients[ind].train_loader.dataset),'size: '+str(len(clients[ind].train_loader.dataset)))

    print("Model: "+str(ARGS.MODEL))
    print("Dataset distribution: "+str(ARGS.DISTRIBUTION)+"  "+str(ARGS.ALPHA)+" ("+str(ARGS.NUM_CLASS_RANGE[0])+','+str(ARGS.NUM_CLASS_RANGE[1])+')')
    print("Number of clients: "+str(ARGS.NUM_CLIENTS))
    print("Number of selected clients: "+str(ARGS.NUM_SELECTED_CLIENTS))
    print("Number of rounds: "+str(ARGS.ROUNDS))
    print("-----------------------------------------")
    
    #INSTANCE SERVER MODEL
    server_model, server_optimizer = get_net_and_optimizer()
    server_criterion = nn.CrossEntropyLoss()
    server_model = server_model.to(ARGS.DEVICE)
    
    #SERVER MODEL -> CLIENTS
    clients=send_server_model_to_clients(server_model, clients)
    
    for i in range(ARGS.ROUNDS):
      start_time = time.time()
    
      #SELECT CLEINTS
      selected_clients=select_clients(clients)
    
      #TRAIN CLIENTS
      selected_clients=train_clients(selected_clients)
    
      #CLIENTS UPDATES -> SERVER & AVERAGE
      server_model = send_client_updates_to_server_and_aggregate(server_model,selected_clients)
    
      #DEBUG: print size,sum_weights,sum_updates for each client
      debug=0
      if(debug):
        print("")
        print_weights(selected_clients,server_model)
    
      #SERVER MODEL -> CLIENTS
      clients=send_server_model_to_clients(server_model, clients)
    
      #TEST
      test_loss, server_accuracy = evaluate(server_model, test_loader)
      server_accuracy=round(server_accuracy,3)
      w_accuracy=round(weighted_accuracy(clients),3)
      seconds=str(round(float(time.time() - start_time),2))
      print(f'After round {i+1:2d} \t server accuracy: {server_accuracy:.3f} \t weighted accuracy: {w_accuracy:.3f} \t time: {seconds} sec.')
    
    print("-----------------------------------------")
    print(f'Final Accuracy of Main Model Federated Learning: {server_accuracy:.3f}')
    print(f'Final Weighted Accuracy of Federated Learning: {w_accuracy:.3f}')
    print("//////////////////////////////////////////////////////////////////")
    print("")
