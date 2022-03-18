import torch.nn as nn
from statistics import mean 
import torch.nn.functional as func
import torch.hub
from options import ARGS
import torchvision.models as m
import torch.optim

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
        
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, ARGS.NUM_CLASSES)
            )
        

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class LeNet5_mod(nn.Module):
    def __init__(self):
        super(LeNet5_mod, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=5))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=5))
        
        if(ARGS.BATCH_NORM==True):
            self.conv1.add_module('1',nn.BatchNorm2d(64)) #conv1.1
            self.conv2.add_module('1',nn.BatchNorm2d(64)) #conv2.1

        
        self.classifier = nn.Sequential(
            nn.Linear(64*5*5, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, ARGS.NUM_CLASSES)
            )

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CNNCifar(nn.Module): #https://www.tensorflow.org/tutorials/images/cnn
    def __init__(self):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = func.relu(self.conv3(x))
        x = self.classifier(x)
        return x



def get_net_and_optimizer():
      if(ARGS.MODEL=="LeNet5"):
          model = LeNet5()
      elif(ARGS.MODEL=="LeNet5_mod"):
          model = LeNet5_mod()
      elif(ARGS.MODEL=="mobilenet_v3_small"):
          model = m.mobilenet_v3_small(pretrained=ARGS.PRETRAIN)
      elif(ARGS.MODEL=="resnet18"):
          model = m.resnet18(pretrained=ARGS.PRETRAIN)
      elif(ARGS.MODEL=="densenet121"):
          model = m.densenet121(pretrained=ARGS.PRETRAIN)
      elif(ARGS.MODEL=="googlenet"):
          model = m.googlenet(pretrained=ARGS.PRETRAIN, aux_logits=False)
      elif(ARGS.MODEL=="CNNCifar"):
          model = CNNCifar()
      
        
      #SET LAST LAYER OUTPUT=NUM_CLASSES
      classifier_name=list(dict(model.named_modules()).keys())[len(list(dict(model.named_modules()).keys()))-1].split(".")[0]
      classifier=getattr(model, classifier_name)
      if isinstance(classifier, nn.Sequential):
          classifier[len(classifier)-1] = nn.Linear(classifier[len(classifier)-1].in_features, ARGS.NUM_CLASSES)
      elif isinstance(classifier, nn.Linear):
          classifier = nn.Linear(classifier.in_features, ARGS.NUM_CLASSES)
      setattr(model, classifier_name, classifier)
      
      
      #FREEZING CONVOLUTIONAL LAYERS
      parameters_to_optimize=model.parameters()
      if(ARGS.FREEZE): 
          for param in model.parameters():
              param.requires_grad = False

          for param in classifier.parameters():
              param.requires_grad = True
          parameters_to_optimize=classifier.parameters()
          
      optimizer = torch.optim.SGD(parameters_to_optimize, lr=ARGS.LR, momentum=ARGS.MOMENTUM)
      return model, optimizer
       
    
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
