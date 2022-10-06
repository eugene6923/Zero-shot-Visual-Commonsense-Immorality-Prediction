import torchvision
import torch
from torch.utils.data import TensorDataset,DataLoader
from torch import nn 
from text_train import get_probs
from models.model import ClassificationHead2
import pandas as pd
import os
import numpy as np
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
cache={}
image_dir='../imagenet/train'
save_dir='../results'
lisdir=os.listdir(image_dir)
y=[]
contents=[]
files=[]
image_class=[]
for index,i in enumerate(lisdir) :
    for j in os.listdir(os.path.join(image_dir,i)) :
        image_class.append(i)
        files.append(os.path.join(image_dir,i,j))
        contents.append(j)
        y.append(0)
    # if index==99 :
    #     break

model='ViT-B/32'
classification2=nn.DataParallel(ClassificationHead2(0.3).to(device), device_ids = [0,1,2,3])
classification =torch.load('./final_log/original_ViT-B32_0.002_64_100.pkl')
classification2.load_state_dict(classification)


# device = torch.device("cuda:0")
model,preprocess2 = clip.load(model, device)

image_features=[]

for i in files :
    # print("image_features :", os.path.split(i)[1] )
    i = Image.open(i)
    image_input = preprocess2(i).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features.append(model.encode_image(image_input))

x=[]
for i in image_features :
    x.append(torch.tensor(i).squeeze(dim=1))
        
x=torch.tensor([item.cpu().detach().numpy() for item in x]).cuda() 
y=torch.tensor(y)

# print(len(x))
data=TensorDataset(x,y) 
data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=64,
                                          shuffle=False)
prob=get_probs(classification2,data_loader)
cache['content']=contents
cache['class']=image_class
cache['prob']=prob.tolist()

cache_data=pd.DataFrame(cache)
print(cache_data.head())
cache_data.to_csv(os.path.join(save_dir,'imagenet_train2.csv'),encoding='utf-8')

cache_data=pd.read_csv("../imagenet/results/imagenet_train2.csv")
filter_data=cache_data[cache_data['prob']>=0.9]
print(filter_data.head())
for i in range(filter_data.shape[0]) : 

    image_class=filter_data['class'].tolist()
    contents=filter_data['content'].tolist()
    img2=Image.open(os.path.join(image_dir,image_class[i],contents[i]))
    img2.save(os.path.join(save_dir,'train2_0.9',os.path.splitext(contents[i])[0]+'.jpeg'))