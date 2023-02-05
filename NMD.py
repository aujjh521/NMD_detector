#import package
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

#define critical file path
pretrained_weight_path = './saved_model/trained_model.pth'

#copy NN from train.py for load model
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

#read trained weight, 先嘗試製造出NMD score
#ref: https://arxiv.org/abs/2104.11408
model = CNN()
model.load_state_dict(torch.load(pretrained_weight_path,map_location=torch.device("cpu")))
print(model)

#測試用的image前處理
test_transform = transforms.Compose([transforms.Resize((128, 128)),
                    transforms.ToTensor()
                    ])

test_image = './local-filename.jpg'
img = Image.open(test_image)
img = test_transform(img).unsqueeze(0) #增加一個維度for batch
print(f'test image size is {img.shape}')

#直接把img丟到model看結果
print(model(img))