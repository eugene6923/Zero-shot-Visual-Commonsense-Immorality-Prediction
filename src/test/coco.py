import glob
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append('../.')
import torch
from clip import load
from PIL import Image
from models.model import ClassificationHead3,ClassificationHead2
from torch.utils.data import TensorDataset,DataLoader
from text_train import evaluate, get_probs,evaluate2
import torch
from torch import nn
import pandas as pd
import argparse

def main(args,dropout):
    cache={}
    y = []
    files = []
    contents=[]
    i=0
    image_class=[]       
    image_features=[]

    i1=0
    i2=0
    coco='./coco/val2017'
    for content in os.listdir(coco) :
        try :
            files.append(os.path.join(coco,content))
            contents.append(content)
            
        except TypeError :
            i1+=1
            print(j)
            pass
        i2+=1
        # if i2 > 2200 :
        #     break
    print("There is {} files that cannot predict in coco".format(i1))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device("cuda:0")
    model,preprocess2 = load(args.model, device)
    
    for i in files:
        try:
            i = Image.open(i)
            image_input = preprocess2(i).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features.append(model.encode_image(image_input))
            y.append(0)
        except TypeError :
            pass

    cache['content']=contents
    cache['label']=y
    cache['immoral']=y

    if args.model in ['ViT-L/14'] :
        classification2=nn.DataParallel(ClassificationHead2(dropout).to(device),device_ids=[0,1,2,3])
        classification =torch.load(args.path)
        classification2.load_state_dict(classification)
    elif args.model in ['ViT-B/32', 'ViT-B/16']:
        classification2=nn.DataParallel(ClassificationHead3(dropout).to(device),device_ids=[0,1,2,3])
        classification =torch.load(args.path)
        classification2.load_state_dict(classification)

    x=[]
    for i in image_features :
        x.append(torch.tensor(i).squeeze(dim=1))
        
    x=torch.tensor([item.cpu().detach().numpy() for item in x]).cuda() 
    y=torch.tensor(y)

    data=TensorDataset(x,y)
    test_dataloader = DataLoader(data, batch_size=64, shuffle=False)
    accuracy=evaluate(classification2,test_dataloader)
    precision, recall,F1=evaluate2(classification2,test_dataloader)
    prob=get_probs(classification2,test_dataloader)
    cache['prob']=prob.tolist()

    if args.save :
        with open(os.path.join(args.save_dir,"coco_prec_recall_acc_total.txt"), "a") as f:
            f.write("\n")
            f.write("category: coco image_model: {} saved_model: {} test acc: {:.3f} recall {} precision {} F1{}".format(args.model,os.path.splitext(os.path.split(args.path)[1])[0],accuracy,recall,precision,F1))

        # cache_data=pd.DataFrame(cache)
        # print(cache_data.head())
        # cache_data.to_csv(os.path.join(args.save_dir,'crawling_coco_{}.csv'.format(os.path.splitext(os.path.split(args.path)[1])[0])),encoding='utf-8')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="ViT-B/32")
    parser.add_argument("--image_dir","-i",default="../coco/val2017")
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--save_dir",type=str,default="../coco/result")
    parser.add_argument("--path","-p", type=str, default="../final_log/original_ViT-B32_0.002_64_100.pkl")
    parser.add_argument("--find","-f",type=bool,default=False)
    args = parser.parse_args()
    
    if args.find == False:
        if args.model.replace("/","") not in args.path :
            raise ValueError
        else :
            main(args,0.5)

    else :

        path="../final_log"
        for i in os.listdir(path) :
            args.path=os.path.join(path,i)
            j=os.path.splitext(i)[0]
            model=j.split('_')[1]
            if model in ['ViT-B16','ViT-B32','ViT-L14'] :
                args.model= model[:5]+'/'+model[5:]

            else :
                args.model=model

            print(args)
            main(args,0.5)