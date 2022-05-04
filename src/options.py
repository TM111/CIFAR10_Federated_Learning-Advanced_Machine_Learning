import argparse
import sys

def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--DEVICE', type=str, default='cuda') #cuda, cpu
    parser.add_argument('--ALGORITHM', type=str, default='FedAvg') #FedAvg, FedAvgM, FedSGD, FedProx, FedNova, SCAFFOLD
    
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--NUM_EPOCHS', type=int, default=3)
    parser.add_argument('--NUM_CLASSES', type=int, default=10)
    parser.add_argument('--BATCH_SIZE', type=int, default=64)
    parser.add_argument('--SERVER_LR', type=float, default=1)
    parser.add_argument('--MU', type=float, default=0.3)
    parser.add_argument('--LR', type=float, default=0.01)
    parser.add_argument('--MOMENTUM', type=float, default=0.9)
    parser.add_argument('--SERVER_MOMENTUM', type=float, default=0.9)
    parser.add_argument('--WEIGHT_DECAY', type=float, default=4e-4)
    
    parser.add_argument('--OPTIMIZER', type=str, default='sgd') #sgd, adam
    parser.add_argument('--PRETRAIN', type=int, default=0)
    parser.add_argument('--FREEZE', type=int, default=0)
    
    parser.add_argument('--FEDIR', type=int, default=0)
    parser.add_argument('--FEDVC', type=int, default=0)
    parser.add_argument('--NVC', type=int, default=256)  #NVC value for FedVC
    
    parser.add_argument('--BATCH_NORM', type=int, default=0)
    parser.add_argument('--GROUP_NORM', type=int, default=0)
    
    parser.add_argument('--CENTRALIZED_MODE', type=int, default=0) # set 1 for standard centralized approach
    parser.add_argument('--DISTRIBUTION', type=str, default='iid') #iid, non_iid, dirichlet, multimodal
    
    parser.add_argument('--ALPHA', type=float, default=0.5)  #[0, 0.05, 0.1, 0.20, 0.5, 1, 10, 100] alpha values for Dirichlet distribution 
    
    parser.add_argument('--RATIO', type=float, default=0.75) #ratio value for Multimodal distribution
    parser.add_argument('--Z', type=int, default=3)    #Z value for Multimodal distribution
    
    parser.add_argument('--NUM_CLASS_RANGE', type=list, default=[1,7])  #maximum and minimum number of classes per client in the non_iid distribution
    
    parser.add_argument('--NUM_CLIENTS', type=int, default=100)
    parser.add_argument('--NUM_SELECTED_CLIENTS', type=int, default=25)
    parser.add_argument('--ROUNDS', type=int, default=25)
    parser.add_argument('--COUNT', type=int, default=0)    
    


    # model arguments
    parser.add_argument('--MODEL', type=str, default='mlp', help='model name') #LeNet5, LeNet5_mod, CNNCifar, CNNNet, AllConvNet, mobilenet_v3_small, resnet18, densenet121, googlenet
    
    parser.add_argument('--COLAB', type=int, default=0) #set to 1 if you want to run this code on Google Colab

    args = parser.parse_args()
    return args

ARGS=args_parser() # global variable


models=["LeNet5","LeNet5_mod","CNNCifar","CNNNet","AllConvNet","mobilenet_v3_small","resnet18","densenet121","googlenet"]
algorithms=['FedAvg', 'FedAvgM', 'FedSGD', 'FedProx', 'FedNova', 'SCAFFOLD']
distributions=['iid', 'non_iid', 'dirichlet', 'multimodal']
alphas=[0, 0.05, 0.1, 0.20, 0.5, 1, 10, 100]

def check_arguments():
    #set default arguments
    if(ARGS.CENTRALIZED_MODE):
        ARGS.ALGORITHM='FedAvg'
        ARGS.DISTRIBUTION='iid'
        ARGS.SERVER_LR=1
        ARGS.NUM_CLIENTS=1
        ARGS.NUM_SELECTED_CLIENTS=1
        ARGS.FEDIR=False
        ARGS.FEDVC=False
        
    errors=''
    errors_count=0
    
    if(ARGS.MODEL not in models):
        errors_count+=1
        errors += str(errors_count)+') Model must be one of them: '
        for m in models: errors += m+', '
        errors = errors[:-2]+'.\n\n'
        
    if(ARGS.ALGORITHM not in algorithms):
        errors_count+=1
        errors += str(errors_count)+') Algorithm must be one of them: '
        for a in algorithms: errors += a+', '
        errors = errors[:-2]+'.\n\n'
        
    if(ARGS.DISTRIBUTION not in distributions):
        errors_count+=1
        errors += str(errors_count)+') Distribution must be one of them: '
        for d in distributions: errors += d+', '
        errors = errors[:-2]+'.\n\n'
    
    if(ARGS.DISTRIBUTION == 'dirichlet' and ARGS.NUM_CLIENTS!=100):
        errors_count+=1
        errors += str(errors_count)+') For dirichlet distribution NUM_CLIENTS must be 100.\n\n'
    
    if(ARGS.DISTRIBUTION == 'dirichlet' and ARGS.ALPHA not in alphas):
        errors_count+=1
        errors += str(errors_count)+') Alpha must be one of them: '
        for a in alphas: errors += str(a)+', '
        errors = errors[:-2]+'.\n\n'
    
    if(ARGS.DISTRIBUTION == 'multimodal' and (ARGS.RATIO<0 or ARGS.RATIO>1)):
        errors_count+=1
        errors += str(errors_count)+') Ratio must be between 0 and 1.\n\n'
    
    if(ARGS.DISTRIBUTION == 'multimodal' and (ARGS.Z<1 or ARGS.Z>4)):
        errors_count+=1
        errors += str(errors_count)+') Z must be between 1 and 4.\n\n'
    
    if(ARGS.MODEL in ["LeNet5","LeNet5_mod","CNNCifar","CNNNet","AllConvNet"]):
        if(ARGS.BATCH_NORM+ARGS.GROUP_NORM>1):
            errors_count+=1
            errors += str(errors_count)+') Disable one of these: BATCH_NORM and GROUP_NORM.\n\n'
        
    if(errors_count>0):
        sys.exit(errors)
    
    #set default arguments
    if(ARGS.ALGORITHM == 'FedSGD'): 
        ARGS.NUM_EPOCHS=1
        ARGS.BATCH_SIZE=10000
    
    if(ARGS.ALGORITHM != 'FedAvgM'): 
        ARGS.SERVER_MOMENTUM=0
        
    if(ARGS.ALGORITHM in ['FedProx', 'FedNova', 'SCAFFOLD']):
        ARGS.FEDIR=0
        ARGS.FEDVC=0
        
    if(ARGS.MODEL in ["LeNet5","LeNet5_mod","CNNCifar","CNNNet","AllConvNet"]):
        ARGS.PRETRAIN=False
        ARGS.FREEZE=False
    
    if(ARGS.MODEL in ["mobilenet_v3_small","resnet18","densenet121","googlenet"]):
        ARGS.BATCH_NORM=0
        ARGS.GROUP_NORM=0
   
