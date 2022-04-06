import torch.nn as nn
from statistics import mean 
import torch.nn.functional as func
import torch.nn.functional as F
import torch.hub
from options import ARGS
import importlib
import torch.optim

#DEFINE NETWORKS
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.conv1=nn.Conv2d(3, 6, kernel_size=5)
        self.conv2=nn.Conv2d(6, 16, kernel_size=5)
        
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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        
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

class CNNNet(nn.Module):  #https://github.com/simoninithomas/cifar-10-classifier-pytorch/blob/master/PyTorch%20Cifar-10%20Classifier.ipynb
    def __init__(self):
        super(CNNNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2,2)
        
        # FC layers
        # Linear layer (64x4x4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        
        # Linear Layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(func.elu(self.conv1(x)))
        x = self.pool(func.elu(self.conv2(x)))
        x = self.pool(func.elu(self.conv3(x)))
        
        # Flatten the image
        x = x.view(-1, 64*4*4)
        x = self.dropout(x)
        x = func.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



class AllConvNet(nn.Module):
    #https://arxiv.org/pdf/1412.6806.pdf
    #https://github.com/StefOe/all-conv-pytorch/blob/master/allconv.py

    def __init__(self,  c=64):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, c, 3)
        self.conv2 = nn.Conv2d(c, c, 3)
        self.conv3 = nn.Conv2d(c, c, 3,stride=2)
        self.conv4 = nn.Conv2d(c, c*2, 3)
        self.conv5 = nn.Conv2d(c*2, c*2, 3)
        self.conv6 = nn.Conv2d(c*2, c*2, 3,stride=2)
        self.conv7 = nn.Conv2d(c*2, c*2, 3)
        self.conv8 = nn.Conv2d(c*2, c*2, 1)
        self.class_conv = nn.Conv2d(c*2, ARGS.NUM_CLASSES, 1)

    def forward(self, x):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv4_out = F.relu(self.conv4(conv3_out))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv7_out = F.relu(self.conv7(conv6_out))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = class_out.reshape(class_out.size(0), class_out.size(1), -1).mean(-1)
        return pool_out


def add_batch_norm(model):
    layers=list(dict(model.named_modules()).keys())
    for layer in layers:
        try:
            new_layer=getattr(model, layer)
        except: continue
    
        if isinstance(new_layer, nn.Conv2d):
            new_layer=nn.Sequential(new_layer,nn.BatchNorm2d(new_layer.out_channels))
            setattr(model, layer, new_layer)
            
        elif isinstance(new_layer, nn.Sequential):
            new_sequential=[]
            for module in new_layer:
                new_sequential.append(module)
                if isinstance(module, nn.Conv2d):
                    new_sequential.append(nn.BatchNorm2d(module.out_channels))
            new_layer = nn.Sequential(*new_sequential)
            setattr(model, layer, new_layer)
    return model
            

def get_net_and_optimizer():
      small_nets=["LeNet5","LeNet5_mod","CNNCifar","CNNNet","AllConvNet"]
      large_nets=["mobilenet_v3_small","resnet18","densenet121","googlenet"] 
        
      if(ARGS.MODEL in small_nets):
          model = globals()[ARGS.MODEL]()
          parameters_to_optimize=model.parameters()
               
      elif(ARGS.MODEL in large_nets):
          if(ARGS.MODEL == 'googlenet'): 
              model= getattr(importlib.import_module('torchvision.models'),'googlenet')(pretrained=ARGS.PRETRAIN,aux_logits=False)
          else: 
              model= getattr(importlib.import_module('torchvision.models'),ARGS.MODEL)(pretrained=ARGS.PRETRAIN)
           
          #SET LAST LAYER OUTPUT=NUM_CLASSES
          classifier_name=list(dict(model.named_modules()).keys())[len(list(dict(model.named_modules()).keys()))-1].split(".")[0]
          classifier=getattr(model, classifier_name)
          if isinstance(classifier, nn.Sequential):
              if isinstance(classifier[len(classifier)-1], nn.Linear):
                  classifier[len(classifier)-1] = nn.Linear(classifier[len(classifier)-1].in_features, ARGS.NUM_CLASSES)
          elif isinstance(classifier, nn.Linear):
              classifier = nn.Linear(classifier.in_features, ARGS.NUM_CLASSES)
          setattr(model, classifier_name, classifier)
              
         
      #ADD BATCH NORMALIZATION
      if(ARGS.BATCH_NORM): 
          model=add_batch_norm(model)
      
      #FREEZING CONVOLUTIONAL LAYERS
      parameters_to_optimize=model.parameters()
      if(ARGS.FREEZE): 
          for param in model.parameters():
              param.requires_grad = False
 
          for param in classifier.parameters():
              param.requires_grad = True
          parameters_to_optimize=classifier.parameters()
          
      if(ARGS.OPTIMIZER=="sgd"):
          optimizer = torch.optim.SGD(parameters_to_optimize, lr=ARGS.LR, momentum=ARGS.MOMENTUM, weight_decay=ARGS.WEIGHT_DECAY)
      elif(ARGS.OPTIMIZER=="adam"):
          optimizer = torch.optim.Adam(parameters_to_optimize, lr=ARGS.LR, weight_decay=ARGS.WEIGHT_DECAY)
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
