import torch.nn as nn
from statistics import mean 
import torch.nn.functional as func
import torch.hub
from options import ARGS

'''
lenet5
resnet18

pretrained ad freeze (transferlearning):
mobilenet
googlenet
'''

#DEFINE NETWORKS
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(3, 6, kernel_size=5))
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=5))
        
        if(ARGS.BATCH_NORM==True):
            self.conv1.add_module('1',nn.BatchNorm2d(6)) #conv1.1
            self.conv2.add_module('1',nn.BatchNorm2d(16)) #conv2.1
            
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, ARGS.NUM_CLASSES)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet5_mod(nn.Module):
    def __init__(self):
        super(LeNet5_mod, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=5))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=5))
        
        if(ARGS.BATCH_NORM==True):
            self.conv1.add_module('1',nn.BatchNorm2d(64)) #conv1.1
            self.conv2.add_module('1',nn.BatchNorm2d(64)) #conv2.1
            
        self.fc1 = nn.Linear(64*5*5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, ARGS.NUM_CLASSES)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def mobilenetV2(pretrain=True):
  model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=pretrain)
  return model





def get_net():
  if(ARGS.MODEL=="LeNet5"):
      return LeNet5()
  if(ARGS.MODEL=="LeNet5_mod"):
      return LeNet5_mod()
  if(ARGS.MODEL=="mobilenetV2"):
      return mobilenetV2()

#DEFINE TEST FUNCTION
def evaluate(net, dataloader):
  criterion = nn.CrossEntropyLoss()
  with torch.no_grad():
    net = net.to(ARGS.DEVICE) # this will bring the network to GPU if DEVICE is cuda
    net.train(False) # Set Network to evaluation mode
    running_corrects = 0
    #iterable = tqdm(dataloader) if print_tqdm else dataloader
    losses = []
    for images, labels in dataloader: 
      images = images.to(ARGS.DEVICE)
      labels = labels.to(ARGS.DEVICE)
      # Forward Pass
      outputs = net(images)
      loss = criterion(outputs, labels)
      losses.append(loss.item())
      # Get predictions
      _, preds = torch.max(outputs.data, 1)
      # Update Corrects
      running_corrects = running_corrects + torch.sum(preds == labels.data).data.item()
    # Calculate Accuracy
    accuracy = running_corrects / float(len(dataloader.dataset))
  return mean(losses),accuracy
