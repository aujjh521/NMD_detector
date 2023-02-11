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

#取出training階段產生的batch norm layer的 mean
training_conv1_batch_norm_mean = model.conv1[1].running_mean
training_conv2_batch_norm_mean = model.conv2[1].running_mean
training_conv3_batch_norm_mean = model.conv3[1].running_mean
NMD_training_batch_norm_vector = torch.cat([training_conv1_batch_norm_mean,
                                            training_conv2_batch_norm_mean,
                                            training_conv3_batch_norm_mean])

#註冊hook (註冊要在model執行forward pass之前)
#依照paper描述, NMD計算會需要取出input testing image 經過conv之後的embeding
#training階段產生的batch norm layer的 mean可以直接透過 running_mean 這個atrribute獲取
#for register_forward_hook (extract NN internal layer output)
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
model.conv1[0].register_forward_hook(get_activation('conv1'))
model.conv2[0].register_forward_hook(get_activation('conv2'))
model.conv3[0].register_forward_hook(get_activation('conv3'))

#測試用的image前處理
test_transform = transforms.Compose([transforms.Resize((128, 128)),
                    transforms.ToTensor()
                    ])

test_image = './local-filename.jpg'
img = Image.open(test_image)
img = test_transform(img).unsqueeze(0) #增加一個維度for batch
print(f'test image size is {img.shape}')

#直接把img丟到model看結果
print(f'classifier output for test image is {model(img)}')

#取出test image的conv1 output (要取mean變成可以跟batch norm的維度比較)
test_conv1 = torch.mean(activation['conv1'], dim=[0,2,3])
test_conv2 = torch.mean(activation['conv2'], dim=[0,2,3])
test_conv3 = torch.mean(activation['conv3'], dim=[0,2,3])
NMD_test_conv_vector = torch.cat([test_conv1,
                              test_conv2,
                              test_conv3])

#把test image產生的conv平均後的串接向量 跟 training的 batch norm mean串接向量 對減
NMD_score_vector = NMD_test_conv_vector - NMD_training_batch_norm_vector
print(f'get NMD_score_vector, shape is {NMD_score_vector.shape}')


