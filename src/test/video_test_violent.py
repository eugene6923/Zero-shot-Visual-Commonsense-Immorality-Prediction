import glob
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append('./')
# print(sys.path)
import torch
import clip
from PIL import Image
from models.model import ClassificationHead3,ClassificationHead2
from torch.utils.data import TensorDataset,DataLoader
from text_train import evaluate, get_probs, evaluate2
import torch
from torch import nn
import pandas as pd
import pickle
import argparse
import pandas as pd


def main(args,dropout):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.deep != True:
        if args.model in ['ViT-B/32', 'ViT-B/16']:
            classification2=nn.DataParallel(ClassificationHead3(dropout).to(device),device_ids=[0,1,2,3])
            classification =torch.load(args.path)
            classification2.load_state_dict(classification)
        elif args.model in ['ViT-L/14']:
            classification2=nn.DataParallel(ClassificationHead3(dropout).to(device),device_ids=[0,1,2,3])
            classification =torch.load(args.path)
            classification2.load_state_dict(classification)

    device2 = torch.device("cuda:3")
    model,preprocess2 = clip.load(args.model, device2)

    image_features=[]

    cache={}
    y = []
    files = []
    contents=[]
    classes=[]
    video_name=[]
    frame=[]
    for content in os.listdir(args.image_dir) :
        for folder in os.listdir(os.path.join(args.image_dir,content)):
            for j in os.listdir(os.path.join(args.image_dir,content,folder)) :
                frame.append(int(j[5:][:-4]))

                img_try = Image.open(os.path.join(args.image_dir,content,folder,j))
                image_input = preprocess2(img_try).unsqueeze(0).to(device2)

                with torch.no_grad():
                    image_features.append(torch.tensor(model.encode_image(image_input),device=device2).squeeze(dim=1))
                if content=="NonViolence":
                    y.append(0)
                else:
                    y.append(1)
                
                classes.append(content)
                video_name.append(folder)
                files.append(os.path.join(args.image_dir,content,folder,j))
                contents.append(j)
            
    print("There is {} files that cannot predict".format(i))
    cache['class']=classes
    cache['frame']=frame
    cache['name']=video_name
    y = np.array(y)
    cache['label']=y
    cache['content']=contents
    x=torch.tensor([item.cpu().detach().numpy() for item in image_features]).cuda() 
    y=torch.tensor(y)

    data=TensorDataset(x,y)
    test_dataloader = DataLoader(data, batch_size=64, shuffle=False)
    accuracy=evaluate(classification2,test_dataloader)
    precision,recall,F1=evaluate2(classification2,test_dataloader)
    prob=get_probs(classification2,test_dataloader)
    cache['prob']=prob.tolist()
    cache_data=pd.DataFrame(cache)
    
    cor1=0
    cor2=0

    total2=0 
    group=cache_data[['label','name','prob']].groupby(['name'], as_index=False).mean()
    total = group.shape[0]
    total1 = group[group['label']==1].shape[0]

    for video in group['name'] :
        if group[group['name']==video]['prob'].values[0]>=args.probability :
            guess=1
            total2+= 1
            if guess==group[group['name']==video]['label'].values[0]:
                cor2+=1
        else: 
            guess=0
        if guess==group[group['name']==video]['label'].values[0]:
            cor1+=1
    
    acc1=cor1/total
    rec1=cor2/total1
    prec1=cor2/total2

    if args.save :
        with open(os.path.join(args.save_dir,"metrics.txt"), "a") as f:
            f.write("\n")
            f.write("video_model: {} probability {} video acc {:.3f} video rec: {:.3f} video prec: {:.3f}".format(args.model,args.probability,acc1,rec1,prec1))

        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="ViT-B/32")
    parser.add_argument("--image_dir","-i",default="./image_dir")
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--save_dir",type=str,default="./save_dir")
    parser.add_argument("--path","-p", type=str, default="../final_log/original_ViT-B32_0.002_64_100.pkl")
    parser.add_argument("--find","-f",type=bool,default=False)
    parser.add_argument("--deep","-d",type=bool,default=False)
    parser.add_argument("--probability",type=bool,default=0.7)
    args = parser.parse_args()
    
    if args.find == False:
        main(args,0.5)

    else :
        path='models path'

        for i in os.listdir(path) :
            args.path=os.path.join(path,i)
            j=os.path.splitext(i)[0]
            model=j.split('_')[1]
            if model in ['ViT-B16','ViT-B32','ViT-L14'] :
                args.model= model[:5]+'/'+model[5:]
            else :
                args.model=model
            main(args,0.5)



                
