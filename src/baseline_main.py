from sampling import dirichlet_distribution, cifar_iid, cifar_noniid
from utils import get_dataset,send_server_model_to_clients, select_clients, train_clients, send_client_updates_to_server_and_aggregate, print_weights, weighted_accuracy,get_dataset_distribution
from options import ARGS
from clients import get_clients_list
from models import get_net, evaluate
import torch.optim
import torch.nn as nn
import time
import random
    
if __name__ == '__main__':
    if(ARGS.COLAB==0): #test locally
        ARGS.MODEL='LeNet5'
        ARGS.DEVICE='cpu'
        ARGS.ALGORITHM='FedIR'
        ARGS.SERVER_MOMENTUM=0.5
        ARGS.DISTRIBUTION=1
        ARGS.ALPHA=0.50
        ARGS.CENTRALIZED_MODE=1
        ARGS.NUM_CLIENTS=25
        ARGS.NUM_SELECTED_CLIENTS=1
        
        
    if(ARGS.CENTRALIZED_MODE==1):
        asd=0
    
    train_set, test_set, train_loader, test_loader = get_dataset()
    
    if(ARGS.DISTRIBUTION==1):  # https://github.com/google-research/google-research/tree/master/federated_vision_datasets
      train_user_images=dirichlet_distribution()
    
    elif(ARGS.DISTRIBUTION==2):
      train_user_images=cifar_iid(train_set)
    
    elif(ARGS.DISTRIBUTION==3):  # https://towardsdatascience.com/preserving-data-privacy-in-deep-learning-part-2-6c2e9494398b
      train_user_images=cifar_noniid(train_set)
     
    train_loader_list={}   #TRAIN LOADER DICT
    for user_id in train_user_images.keys():
      dataset_ = torch.utils.data.Subset(train_set, train_user_images[user_id])
      dataloader = torch.utils.data.DataLoader(dataset=dataset_, batch_size=ARGS.BATCH_SIZE, shuffle=False)
      train_loader_list[user_id]=dataloader
      
      
    clients_list=get_clients_list(train_loader_list, train_set, test_set)
    
    
    for i in range(random.randint(2,7)):
      random.shuffle(clients_list)
    print("Distribution of trainset for client 0 ",get_dataset_distribution(clients_list[0].train_loader.dataset))
    print("Distribution of testset for client 0 ",get_dataset_distribution(clients_list[0].test_loader.dataset))
    
    
    print("Model: "+str(ARGS.MODEL))
    print("Dataset distribution: "+str(ARGS.DISTRIBUTION)+"  "+str(ARGS.ALPHA)+" ("+str(ARGS.NUM_CLASS_RANGE[0])+','+str(ARGS.NUM_CLASS_RANGE[1])+')')
    print("Number of clients: "+str(ARGS.NUM_CLIENTS))
    print("Number of selected clients: "+str(ARGS.NUM_SELECTED_CLIENTS))
    print("Number of rounds: "+str(ARGS.ROUNDS))
    print("-----------------------------------------")
    
    #INSTANCE SERVER MODEL
    server_model=get_net()
    server_optimizer = torch.optim.SGD(server_model.parameters(), lr=ARGS.SERVER_LR, momentum=ARGS.MOMENTUM)
    server_criterion = nn.CrossEntropyLoss()
    server_model = server_model.to(ARGS.DEVICE)
    
    #SERVER MODEL -> CLIENTS
    clients_list=send_server_model_to_clients(server_model, clients_list)
    
    for i in range(ARGS.ROUNDS):
      start_time = time.time()
    
      #SELECT CLEINTS
      selected_clients_list=select_clients(clients_list)
    
      #TRAIN CLIENTS
      selected_clients_list=train_clients(selected_clients_list)
    
      #CLIENTS UPDATES -> SERVER & AVERAGE
      server_model= send_client_updates_to_server_and_aggregate(server_model,selected_clients_list)
    
      #DEBUG
      debug=1
      if(debug):
        print("")
        print_weights(selected_clients_list,server_model)
    
      #SERVER MODEL -> CLIENTS
      clients_list=send_server_model_to_clients(server_model, clients_list)

    
      #TEST
      test_loss, main_model_accuracy = evaluate(server_model, server_criterion, test_loader)
      main_model_accuracy=round(main_model_accuracy,3)
      w_accuracy=round(weighted_accuracy(clients_list),3)
      seconds=str(round(float(time.time() - start_time),2))
      print("After round "+str(i+1)+"  main model accuracy: "+str(main_model_accuracy)+"    weighted accuracy: "+str(w_accuracy)+"   time: "+seconds+" sec.")
    
    print("-----------------------------------------")
    print("Final Accuracy of Main Model Federated Learning: "+str(main_model_accuracy))
    print("Final Weighted Accuracy of Federated Learning: "+str(w_accuracy))
    print("//////////////////////////////////////////////////////////////////")
    print("")
