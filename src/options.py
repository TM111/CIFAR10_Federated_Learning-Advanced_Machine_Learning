
import argparse
    
def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--DEVICE', type=str, default='cuda',
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
    parser.add_argument('--LR', type=float, default=0.01,
                        help="number of rounds of training")
    parser.add_argument('--MOMENTUM', type=float, default=0.9,
                        help="number of rounds of training")
    parser.add_argument('--WEIGHT_DECAY', type=float, default=4e-4,
                        help="number of rounds of training")
    
    parser.add_argument('--CENTRALIZED_MODE', type=int, default=0,
                        help="number of rounds of training")
    parser.add_argument('--DISTRIBUTION', type=int, default=1,
                        help="number of rounds of training")
    parser.add_argument('--ALPHA', type=str, default='0.50',
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
    
    parser.add_argument('--COLAB', type=int, default=0, help='run on colab or locally')

    args = parser.parse_args()
    return args