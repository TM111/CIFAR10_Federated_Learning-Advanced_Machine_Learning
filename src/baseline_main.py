
from sampling import dirichlet_distribution, cifar_iid, cifar_noniid
from utils import get_dataset,send_server_model_to_clients, select_clients, train_clients, send_client_models_to_server_and_aggregate, print_weights, weighted_accuracy
from options import args_parser
from clients import get_clients_list
from models import get_net, evaluate
import torch.optim
import torch.nn as nn
import time
import random

args = args_parser()
if(args.COLAB==False):
    args.MODEL='LeNet5'
    args.DEVICE='cpu'
    args.DISTRIBUTION=3
    args.CENTRALIZED_MODE=True

if __name__ == '__main__':
    if(args.COLAB==True):
        args = args_parser()
        
    if(args.CENTRALIZED_MODE):
        args.DISTRIBUTION=1
        args.NUM_CLIENTS=100
        args.NUM_SELECTED_CLIENTS=1
        
    centralized_accuracy=9

    test_set, train_set, train_loader, test_loader =get_dataset(args)
    
    distribution=args.DISTRIBUTION
    alpha=args.ALPHA
    
    if(distribution==1):  # https://github.com/google-research/google-research/tree/master/federated_vision_datasets
      train_user_images=dirichlet_distribution(alpha,args)
    
    elif(distribution==2):
      train_user_images=cifar_iid(train_set,args)
    
    elif(distribution==3):  # https://towardsdatascience.com/preserving-data-privacy-in-deep-learning-part-2-6c2e9494398b
      train_user_images=cifar_noniid(train_set,args)
     
        
    train_loader_list={}   #TRAIN LOADER DICT
    for user_id in train_user_images.keys():
      dataset_ = torch.utils.data.Subset(train_set, train_user_images[user_id])
      dataloader = torch.utils.data.DataLoader(dataset=dataset_, batch_size=args.BATCH_SIZE, shuffle=False)
      train_loader_list[user_id]=dataloader
      
      
    clients_list=get_clients_list(train_loader_list,test_set,args)
    
    
    for i in range(random.randint(2,7)):
      random.shuffle(clients_list)
    ind=0
    l=[0,0,0,0,0,0,0,0,0,0]
    for i in range(len(clients_list[ind].train_loader.dataset)):
      index=clients_list[ind].train_loader.dataset[i][1]
      l[index]=l[index]+1
    print("Distribution of trainset for client "+str(ind),l)
    
    l=[0,0,0,0,0,0,0,0,0,0]
    for i in range(len(clients_list[ind].test_loader.dataset)):
      index=clients_list[ind].test_loader.dataset[i][1]
      l[index]=l[index]+1
    print("Distribution of testset for client  "+str(ind),l)
      
    print("Model: "+str(args.MODEL))
    print("Dataset distribution: "+str(distribution)+"  "+str(alpha)+" ("+str(args.NUM_CLASS_RANGE[0])+','+str(args.NUM_CLASS_RANGE[1])+')')
    print("Number of clients: "+str(args.NUM_CLIENTS))
    print("Number of selected clients: "+str(args.NUM_SELECTED_CLIENTS))
    print("Number of rounds: "+str(args.ROUNDS))
    print("-----------------------------------------")
    
    #INSTANCE MAIN MODEL
    server_model=get_net(args)
    server_optimizer = torch.optim.SGD(server_model.parameters(), lr=args.SERVER_LR, momentum=args.MOMENTUM)
    server_criterion = nn.CrossEntropyLoss()
    server_model = server_model.to(args.DEVICE)
    
    #MAIN MODEL -> CLIENTS
    clients_list=send_server_model_to_clients(server_model, clients_list)
    
    for i in range(args.ROUNDS):
      start_time = time.time()
    
      #SELECT CLEINTS
      round_clients_list=select_clients(clients_list,args)
    
      #TRAIN CLIENTS
      round_clients_list=train_clients(round_clients_list,args)
    
      #CLIENTS -> MAIN MODEL & AVERAGE
      server_model= send_client_models_to_server_and_aggregate(server_model,round_clients_list,args)
    
      #DEBUG
      debug=1
      if(debug):
        print("")
        print_weights(round_clients_list,server_model,args)
    
      #MAIN MODEL -> CLIENTS
      clients_list=send_server_model_to_clients(server_model, clients_list)
    
      #TEST
      test_loss, main_model_accuracy = evaluate(server_model, server_criterion, test_loader,args)
      main_model_accuracy=round(main_model_accuracy,3)
      w_accuracy=round(weighted_accuracy(clients_list,args),3)
      seconds=str(round(float(time.time() - start_time),2))
      print("After round "+str(i+1)+"  main model accuracy: "+str(main_model_accuracy)+"    weighted accuracy: "+str(w_accuracy)+"   time: "+seconds+" sec.")
    
    print("-----------------------------------------")
    print("Final Accuracy of Main Model Federated Learning: "+str(main_model_accuracy))
    print("Final Weighted Accuracy of Federated Learning: "+str(w_accuracy))
    print("Final Accuracy of standard approach: "+str(centralized_accuracy))
    print("//////////////////////////////////////////////////////////////////")
    print("")
