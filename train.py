#%%
#package import
#general
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#pytorch related
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

#pytorch 讀取image好用的功能
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import DatasetFolder


#%%
#觀察後可以知道train資料夾裡面的檔案開頭會標記類別(cat / dog)
#因為train裡面的檔案格式是把所有東西都塞在同一個資料夾
#所以沒辦法直接調用ImageFolder 或是 DatasetFolder(這兩個都需要把檔案依照類別存放不同資料夾)
#比較好的做法應該是先產生一個mapping table, 再從mapping table去自己製造class dataset
import os
mapping_table = pd.DataFrame({'image_name':os.listdir('./input//train')}) #讀取train資料夾底下的所有檔案路徑,並存成dataframe
mapping_table['label'] =mapping_table['image_name'].apply(lambda x: x.split('.')[0])

#依照官方定義的做label encoding(1 = dog, 0 = cat)
label_encode_mapping = {'dog':1,
              'cat':0
              }
mapping_table['label_encode'] =mapping_table['label'].apply(lambda x:label_encode_mapping[x])

mapping_table


#建立transform用的pipeline (train/test拆開,因為train/test的transform步驟可能不一樣)
#一般會在train裡面加一些augmentation
#train
train_transform = transforms.Compose([transforms.Resize((128, 128)),
                    transforms.ToTensor()
                      ])
#test
test_transform = transforms.Compose([transforms.Resize((128, 128)),
                    transforms.ToTensor()
                      ])

#建立class dataset
class DogCatDataset(Dataset):
  def __init__(self, mapping_table, train_data_dir, transform):
    self.mapping_table = mapping_table
    self.train_data_dir = train_data_dir
    self.transform = transform

  def __len__(self):
    return len(self.mapping_table)

  def __getitem__(self, index):
    img_name = os.path.join(self.train_data_dir, self.mapping_table.loc[index,'image_name'])
    #print(img_name)
    img = Image.open(img_name)
    label = self.mapping_table.loc[index,'label_encode']
    if self.transform:
        img = self.transform(img)

    return img , label

train_data_dir = './input//train'
train_dataset = DogCatDataset(mapping_table, train_data_dir, train_transform)


#這邊有個地雷要小心
#使用Image.open讀取出來的data類別是pil image, 而transforms.ToTensor()會把pil轉成tensor
#重點是pil image & tensor在維度的定義上不一樣
#細節參考 https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html#torchvision.transforms.ToTensor
#然後因為matplotlib在做圖的時候也有一組自己預設的讀取維度(plt.imshow), 所以如果想要把經過transforms.ToTensor()的圖正常畫出來,就會需要做一次permute
#但是因為pytorch預設使用的image運算維度就跟transforms.ToTensor()一樣,所以在後續training上不需要再調整(廢話,因為transforms.ToTensor()就是pytorch自己的method,當然會整合好)
'''
img1 = Image.open(os.path.join('./input//train',mapping_table.loc[2,'image_name'])) #pil image
print(f'before transform, the shape is {img1.size}, type {type(img1)}')
trs = transforms.ToTensor()
trs_img1 = trs(img1) #pytorch tensor (要注意維度順序會改變)
print(f'after transform, the shape is {trs_img1.shape}, type {type(trs_img1)}')
plt.imshow(trs_img1.permute((1,2,0)))#把維度交換為做圖(matplotlib)預設定義的
plt.show()
'''

'''
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
  sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
  img, label = train_dataset[sample_idx]
  figure.add_subplot(rows, cols, i)
  plt.title(label)
  plt.axis("off")
  plt.imshow(img.permute((1,2,0))) #為了做圖才會特別需要調整順序
plt.show()
'''

#dataset split into train & validation
'''
#這邊要更精進的話可以改成
1. k-fold cross validation
2. torch.utils.data.random_split裡面改成辨認ratio去拆包
'''
train_data_ratio = 0.9
train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [23000,2000],
        generator=torch.Generator().manual_seed(1)
        )

#%%
#define model (CNN)
#CNN
#ref: https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Sequential(         
        nn.Conv2d(
            in_channels=3,              
            out_channels=64,            
            kernel_size=3,              
            stride=1,                   
            padding=1,                  
        ),
        nn.BatchNorm2d(64),                              
        nn.ReLU(),                      
        nn.MaxPool2d(2, 2, 0),    
    )
    self.conv2 = nn.Sequential(         
        nn.Conv2d(64, 128, 3, 1, 1),  
        nn.BatchNorm2d(128),     
        nn.ReLU(),                      
        nn.MaxPool2d(2, 2, 0),                
    )

    self.conv3 = nn.Sequential(         
        nn.Conv2d(128, 256, 3, 1, 1),  
        nn.BatchNorm2d(256),     
        nn.ReLU(),                      
        nn.MaxPool2d(4, 4, 0),                
    )

    # fully connected layer, output 2 classes
    self.out = nn.Linear(256 * 8*8, 2)
  def forward(self, x):
      #print(f'1. {x.shape}')
      x = self.conv1(x)
      #print(f'2. {x.shape}')
      x = self.conv2(x)
      #print(f'3. {x.shape}')
      #print(x.shape)
      x = self.conv3(x)
      # flatten the output of conv2
      x = x.view(x.size(0), -1) 
      #print(f'4. {x.shape}')
      #print(x.shape)      
      output = self.out(x)
      #print(f'5. {x.shape}')
      return output

#%%
#define training procedure
#如果沒有用gpu會慢到爆炸

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f'device is {device}')

#create dataloader
BATCH_SIZE = 128
train_dataloader = DataLoader(train_subset, shuffle=True, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_subset, shuffle=False, batch_size=BATCH_SIZE)

#hyper parameter
lr = 0.0003
epoch = 20
model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

#用來記錄每個batch的loss & acc
train_loss_record = []
val_loss_record = []
train_acc_record = []
val_acc_record = []

print('start training')
for epo in range(epoch):
  #each epo, initialize loss,acc for training & validation
  train_acc = 0.0
  train_loss = 0.0
  val_acc = 0.0
  val_loss = 0.0

  #training
  model.train()
  for i,(img, target) in enumerate(train_dataloader):

    img = img.to(device)
    target = target.to(device)

    #forward pass
    with amp.autocast():
      y_hat = model(img)
      #print(y_hat, target)
      #print(y_hat.shape, target.shape)
      loss = loss_fn(y_hat,target) # this is the mean loss of all data in one mini batch 
      _, train_pred = torch.max(y_hat, dim=1) # get the index of the class with the highest probability (for acc calculation)

    #backward prop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #append to batch_loss record
    train_loss_record.append(loss.cpu().item())
    train_acc_record.append((train_pred.cpu() == target.cpu()).sum().item())


    #accumulate mini batch loss for training (for epoch)
    train_loss = train_loss + loss.cpu().item()
    train_acc = train_acc + (train_pred.cpu() == target.cpu()).sum().item()
    print(f'finish batch: {i}')

  #validation
  model.eval()
  with torch.no_grad():
    for i,(img, target) in enumerate(val_dataloader):
      img = img.to(device)
      target = target.to(device)

      #forward pass
      y_hat = model(img)
      loss = loss_fn(y_hat,target)
      _, val_pred = torch.max(y_hat, 1) # get the index of the class with the highest probability (for acc calculation)

      #append to batch_loss record
      val_loss_record.append(loss.cpu().item())
      val_acc_record.append((val_pred.cpu() == target.cpu()).sum().item())

      #accumulate mini batch loss for validation
      val_loss = val_loss + loss.cpu().item()
      val_acc = val_acc + (val_pred.cpu() == target.cpu()).sum().item()

  #summirized loss,acc in this epoc (need to normalized by batch number & data point)
  #len(train_data_loader): batch number -> this is used for loss due to in each mini batch loss is the mean value of that mini batch
  #len(train_dataset): number of data point -> this is use for acc due to for acc it is not averaged value in each mini batch
  print(f'epo [{epo+1}/{epoch}]:\ntrain_loss = {train_loss/len(train_dataloader)}, vali_loss = {val_loss/len(val_dataloader)}\n \
  train_acc = {train_acc/len(train_dataset)}, vali_acc = {val_acc/len(val_subset)}')

print('finish training')

#%%
#save trained mode
import torch
save_path = './saved_model/'
torch.save(model.state_dict(), save_path + 'trained_model.pth')
print('save trained modle')

# %%
