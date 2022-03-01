import torch.utils.data
import csv
from os import path
import os
import urllib.request 
import zipfile
import random

def cifar_parser(line, is_train=True):
  if is_train:
    user_id, image_id, class_id = line
    return user_id, image_id, class_id
  else:
    image_id, class_id = line
    return image_id, class_id


def dirichlet_distribution(alpha):    # generate trainset split from csv
  #download csv files
  url="http://storage.googleapis.com/gresearch/federated-vision-datasets/cifar10_v1.1.zip"
  dir="/content/cifar10_csv"
  try:
    os.mkdir(dir)
  except:
    print("Folder already exist")

  urllib.request.urlretrieve(url, "/content/cifar10_csv/cifar.zip")
  with zipfile.ZipFile("/content/cifar10_csv/cifar.zip","r") as zip_ref:
      zip_ref.extractall("/content/cifar10_csv")

  train_file="cifar10_csv/federated_train_alpha_"+alpha+".csv"
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


def cifar_iid(train_set,NUM_CLIENTS): # all clients have all classes with the same data distribution
  user_images={}
  classes_dict={}
  for i in range(len(train_set)):
    label=train_set[i][1]
    if(label not in classes_dict.keys()):
      classes_dict[label]=[]
    classes_dict[label].append(i)
  classes_index=[]
  for label in classes_dict.keys():
    classes_index=classes_index+classes_dict[label]

  count=0
  for i in classes_index:
    if(str(count) not in user_images.keys()):
      user_images[str(count)]=[]
    user_images[str(count)].append(i)
    count=count+1
    if(count==NUM_CLIENTS):
      count=0
  return user_images


def cifar_noniid(train_set,NUM_CLASSES,NUM_CLASS_RANGE): # all clients have a number of class beetwen 1 and 4 with the same data distribution
  user_images=cifar_iid()
  for key in user_images.keys():
    n_classes=random.randint(NUM_CLASS_RANGE[0],NUM_CLASS_RANGE[1])
    list_of_class=random.sample(range(0, NUM_CLASSES), n_classes)
    new_index_list=[]
    for i in user_images[key]:
      label=int(train_set[i][1])
      if(label in list_of_class):
        new_index_list.append(i)
    user_images[key]=new_index_list
  return user_images


def generated_test_distribution(classes, test_set, num_samples,BATCH_SIZE):    # generate testset with specific class and size
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
      if(len(test_user_images)==int(num_samples/len(classes)+1)*count):
        break
  dataset_ = torch.utils.data.Subset(test_set, test_user_images)
  dataloader = torch.utils.data.DataLoader(dataset=dataset_, batch_size=BATCH_SIZE, shuffle=False)
  return dataloader