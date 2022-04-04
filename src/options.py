import argparse
import sys

def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--DEVICE', type=str, default='cuda',
                        help="number of rounds of training")
    parser.add_argument('--ALGORITHM', type=str, default='FedAvg',
                        help="number of rounds of training")
    
    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--NUM_EPOCHS', type=int, default=1,
                        help="number of rounds of training")
    parser.add_argument('--NUM_CLASSES', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--BATCH_SIZE', type=int, default=64,
                        help="number of rounds of training")
    parser.add_argument('--SERVER_LR', type=int, default=1,
                        help="number of rounds of training")
    parser.add_argument('--MU', type=float, default=0,
                        help="number of rounds of training")
    parser.add_argument('--LR', type=float, default=0.01,
                        help="number of rounds of training")
    parser.add_argument('--MOMENTUM', type=float, default=0.9,
                        help="number of rounds of training")
    parser.add_argument('--SERVER_MOMENTUM', type=float, default=0,
                        help="number of rounds of training")
    parser.add_argument('--WEIGHT_DECAY', type=float, default=4e-4,
                        help="number of rounds of training")
    
    parser.add_argument('--OPTIMIZER', type=str, default='sgd',
                        help="number of rounds of training")
    parser.add_argument('--PRETRAIN', type=int, default=0,
                        help="number of rounds of training")
    parser.add_argument('--FREEZE', type=int, default=0,
                        help="number of rounds of training")
    
    parser.add_argument('--FEDIR', type=int, default=0,
                        help="number of rounds of training")
    parser.add_argument('--FEDVC', type=int, default=0,
                        help="number of rounds of training")
    parser.add_argument('--NVC', type=int, default=200,
                        help="number of rounds of training")
    
    parser.add_argument('--BATCH_NORM', type=int, default=0,
                        help="number of rounds of training")
    
    parser.add_argument('--CENTRALIZED_MODE', type=int, default=0,
                        help="number of rounds of training")
    parser.add_argument('--DISTRIBUTION', type=str, default='iid',
                        help="number of rounds of training")
    
    parser.add_argument('--ALPHA', type=float, default=0.5,
                        help="number of rounds of training")
    
    parser.add_argument('--RATIO', type=float, default=0.8,
                        help="number of rounds of training")
    parser.add_argument('--Z', type=int, default=2,
                        help="number of rounds of training")
    
    parser.add_argument('--NUM_CLASS_RANGE', type=list, default=[1,7],
                        help="number of rounds of training")
    
    parser.add_argument('--NUM_CLIENTS', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--NUM_SELECTED_CLIENTS', type=int, default=3,
                        help="number of rounds of training")
    parser.add_argument('--ROUNDS', type=int, default=20,
                        help="number of rounds of training")
    


    # model arguments
    parser.add_argument('--MODEL', type=str, default='mlp', help='model name')
    
    parser.add_argument('--COLAB', type=int, default=0,
                        help="number of rounds of training")

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
        
    if(errors_count>0):
        sys.exit(errors)
    
    #set default arguments
    if(ARGS.ALGORITHM == 'FedSGD'): 
        ARGS.NUM_EPOCHS=1
        ARGS.BATCH_SIZE=999999
        
    if(ARGS.ALGORITHM in ['FedProx', 'FedNova', 'SCAFFOLD']):
        ARGS.FEDIR=False
        ARGS.FEDVC=False
        
    if(ARGS.MODEL not in ["LeNet5","LeNet5_mod","CNNCifar","CNNNet","AllConvNet"]):
        ARGS.PRETRAIN=False
        ARGS.FREEZE=False
    
    if(ARGS.MODEL not in ["mobilenet_v3_small","resnet18","densenet121","googlenet"]):
        ARGS.BATCH_NORM=0
        
        
        