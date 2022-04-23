from sampling import get_train_distribution,get_cached_clients
from utils import get_dataset,send_server_model_to_clients, select_clients, train_clients, send_client_updates_to_server_and_aggregate, print_weights, weighted_accuracy,get_dataset_distribution
from options import ARGS, check_arguments
from clients import get_clients_list
from server import Server
from models import evaluate
import time
import random


if __name__ == '__main__':
    if(ARGS.COLAB == 0): #test locally
    
        ARGS.DEVICE='cpu'
        ARGS.MODEL='LeNet5' # LeNet5, LeNet5_mod, CNNCifar, CNNNet, AllConvNet, 
        ARGS.NUM_EPOCHS=2                     # mobilenet_v3_small, resnet18, densenet121, googlenet 
        ARGS.BATCH_NORM=0
        ARGS.PRETRAIN=False
        ARGS.FREEZE=False
        
        ARGS.ALGORITHM='FedAvg'  # FedAvg, FedAvgM, FedSGD, FedProx, FedNova, SCAFFOLD
        
        ARGS.DISTRIBUTION='dirichlet' # iid, non_iid, dirichlet, multimodal
        ARGS.ALPHA=0
        #ARGS.CENTRALIZED_MODE=True
        ARGS.NUM_CLIENTS=100
        ARGS.NUM_SELECTED_CLIENTS=3

    
    #ARGS.LR=0.005
    #ARGS.COUNT=4
    #ARGS.SERVER_MOMENTUM=0.01
    check_arguments() # control arguments
    
    train_set, test_set, train_loader, test_loader = get_dataset()  #download dataset

    #INSTANCE CLIENTS
    Clients=get_cached_clients() #get cached clients list
    if(Clients==None):
        train_loader_list=get_train_distribution(train_set) #split trainset to current distribution
        Clients=get_clients_list(train_loader_list, train_set, test_set) #generate client list
    

    for i in range(random.randint(2,7)):
      random.shuffle(Clients)
    ind=0
    print("Distribution of trainset for client "+str(ind),get_dataset_distribution(Clients[ind].train_loader.dataset),'size: '+str(len(Clients[ind].train_loader.dataset)))
            
    print("-----------------------------------------")
    print("Count:",ARGS.COUNT)
    print("Model:",ARGS.MODEL)
    print("Batch norm:",ARGS.BATCH_NORM)
    print("Freeze:",ARGS.FREEZE)
    print("Pretrain:",ARGS.PRETRAIN)
    
    print("Dataset distribution:",ARGS.DISTRIBUTION)
    if(ARGS.DISTRIBUTION=="dirichlet"):
        print("Alpha: ",ARGS.ALPHA)
    if(ARGS.DISTRIBUTION=="multimodal"):
        print("Ratio: ",ARGS.RATIO,"  Z:",ARGS.Z)
    if(ARGS.DISTRIBUTION=="non_iid"):
        print("Class range: ",ARGS.NUM_CLASS_RANGE)
        
    print("Algorithm:",ARGS.ALGORITHM)
    print("FedIR:",ARGS.FEDIR,"   FedVC:",ARGS.FEDVC)
    if(ARGS.DISTRIBUTION=="FedAvgM"):
        print("Server momentum: ",ARGS.SERVER_MOMENTUM)
    if(ARGS.DISTRIBUTION=="FedProx"):
        print("Mu: ",ARGS.MU)
        
    print("Number of clients:",ARGS.NUM_CLIENTS)
    print("Number of selected clients:",ARGS.NUM_SELECTED_CLIENTS)
    print("Number of rounds:",ARGS.ROUNDS)

    print("-----------------------------------------")
    
    #INSTANCE SERVER
    Server = Server()
    
    #SERVER MODEL -> CLIENTS
    Clients = send_server_model_to_clients(Server, Clients)

    for i in range(ARGS.ROUNDS):
      start_time = time.time()
    
      #SELECT CLEINTS
      selected_clients = select_clients(Clients)
      
      #TRAIN CLIENTS
      selected_clients = train_clients(selected_clients)
      
      #CLIENTS UPDATES -> SERVER & AVERAGE
      Server = send_client_updates_to_server_and_aggregate(Server, selected_clients)
    
      #DEBUG: print size,sum_weights,sum_updates for each client
      debug=1
      if(debug): print_weights(selected_clients, Server.model)
    
      #SERVER MODEL -> CLIENTS
      Clients = send_server_model_to_clients(Server, Clients)
    
      #TEST
      seconds=str(round(float(time.time() - start_time),2))
      server_loss, server_accuracy = evaluate(Server.model, test_loader)
      w_accuracy=weighted_accuracy(Clients)
      print(f'After round {i+1:2d} \t server accuracy: {server_accuracy:.3f} \t server loss: {server_loss:.3f} \t weighted accuracy: {w_accuracy:.3f} \t train time: {seconds} sec.')
    
    print("-----------------------------------------")
    print(f'Final Accuracy of Main Model Federated Learning: {server_accuracy:.3f}')
    print(f'Final Weighted Accuracy of Federated Learning: {w_accuracy:.3f}')
    print("//////////////////////////////////////////////////////////////////")
    print("")
