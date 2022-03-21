import torch.utils.data
import csv
from os import path
import os
import urllib.request 
import zipfile
import random
from options import ARGS
import pickle as pickle

def cifar_parser(line, is_train=True):
  if is_train:
    user_id, image_id, class_id = line
    return user_id, image_id, class_id
  else:
    image_id, class_id = line
    return image_id, class_id


def dirichlet_distribution():    # generate trainset split from csv
  #download csv files
  url="http://storage.googleapis.com/gresearch/federated-vision-datasets/cifar10_v1.1.zip"
  if(ARGS.COLAB):
      dir="/content/cifar10_csv"
  else:
      dir=os.getcwd()+"/cifar10_csv"
  try:
    os.mkdir(dir)
  except:
    print("Folder already exist")
    
  urllib.request.urlretrieve(url, dir+"/cifar.zip")
  with zipfile.ZipFile(dir+"/cifar.zip","r") as zip_ref:
      zip_ref.extractall(dir)
  
  alpha=str("{:.2f}".format(ARGS.ALPHA))
  train_file=dir+"/federated_train_alpha_"+alpha+".csv"
  """Inspects the federated train split."""
  print('Train file: %s' % train_file)
  if not path.exists(train_file):
    print('Error: file does not exist.')
    return
  user_images={}
  with open(train_file) as f:
    reader = csv.reader(f)
    next(reader)  # skip header.
    for line in reader:
      user_id, image_id, class_id = cifar_parser(line, is_train=True)
      if(user_id not in user_images.keys()):
        user_images[user_id]=[]
      user_images[user_id].append(int(image_id))
  return user_images


def cifar_iid(train_set): # all clients have all classes with the same data distribution
  user_images={}
  classes_dict={}
  for i in range(len(train_set)):
    label=train_set[i][1]
    if(label not in classes_dict.keys()):
      classes_dict[label]=[]
    classes_dict[label].append(i)
  classes_index=[]
  for label in classes_dict.keys():
    for i in range(random.randint(2,7)):
      random.shuffle(classes_dict[label])
    classes_index=classes_index+classes_dict[label]

  client_id=0
  for i in classes_index:
    if(str(client_id) not in user_images.keys()):
      user_images[str(client_id)]=[]
    user_images[str(client_id)].append(i)
    client_id=client_id+1
    if(client_id==ARGS.NUM_CLIENTS):
      client_id=0
  return user_images


def cifar_noniid(train_set): # all clients have a number of class beetwen 1 and 4 with the same data distribution
  user_images=cifar_iid(train_set)
  for key in user_images.keys():
    n_classes=random.randint(ARGS.NUM_CLASS_RANGE[0],ARGS.NUM_CLASS_RANGE[1])
    list_of_class=random.sample(range(0, ARGS.NUM_CLASSES), n_classes)
    new_index_list=[]
    for i in user_images[key]:
      label=int(train_set[i][1])
      if(label in list_of_class):
        new_index_list.append(i)
    user_images[key]=new_index_list
  return user_images

def get_train_distribution(train_set):
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
    return train_loader_list

def get_cached_clients():
    if(ARGS.COLAB):
        dir="/content/clients_list_cache/"
    else:
        dir=os.getcwd()+"/clients_list_cache/"
    file=str(ARGS.DISTRIBUTION)+str(ARGS.ALPHA)+str(ARGS.NUM_CLASS_RANGE[0])+str(ARGS.NUM_CLASS_RANGE[1])+str(ARGS.NUM_CLIENTS)
    if(os.path.exists(dir+file)):
        with open(dir+file, 'rb') as config_dictionary_file:
            return pickle.load(config_dictionary_file)
    else:
        files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        for f in files: os.remove(dir+f)
        return None
    
def generated_test_distribution(classes, test_set, num_samples):    # generate testset with specific class and size
  test_user_images=[]
  count=0
  for c in classes:
    count=count+1
    indexes=list(range(len(test_set)))
    for i in range(random.randint(2,7)):
      random.shuffle(indexes)
    for i in indexes:
      if(test_set[i][1]==c):
        test_user_images.append(i)
      if(len(test_user_images)==int(num_samples/len(classes))*count):
        break
  dataset_ = torch.utils.data.Subset(test_set, test_user_images)
  dataloader = torch.utils.data.DataLoader(dataset=dataset_, batch_size=ARGS.BATCH_SIZE)
  return dataloader