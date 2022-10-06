import glob
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append('../')
# print(sys.path)
import torch
import clip
from PIL import Image
from models.model import ClassificationHead3,ClassificationHead2
from torch.utils.data import TensorDataset,DataLoader
from text_train import evaluate, get_probs,evaluate2
import torch
from torch import nn
import pandas as pd
import argparse

def main(args,dropout):
    y = []
    cache={}


    files = []

    sexual_label=pd.read_csv('./SexualIntentDetection-master/Features/SexualIntent.csv')
    sexual_label.columns=['image','sexuality']

    files = []
    contents=[]
    i=0

    
    for content in os.listdir(args.image_dir) :
        try :
            label=float(sexual_label[sexual_label['image']==content.replace('.jpg','').replace('.png','').replace('.jpeg','')]['sexuality'].values)
            y.append(label)
            files.append(os.path.join(args.image_dir,content))
            contents.append(content)
        except TypeError :
            i+=1
            print(content)
            pass
    print("There is {} files that cannot predict".format(i))

    cache['content']=contents
    cache['label']=y
    y = list(1 if x ==1 else 0 for x in y)
    cache['immoral']=y

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model,preprocess2 = clip.load(args.model, device)

    image_features=[]

    for i in files :
        i = Image.open(i)
        image_input = preprocess2(i).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features.append(model.encode_image(image_input))

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
    recall,precision,F1=evaluate2(classification2,test_dataloader)
    prob=get_probs(classification2,test_dataloader)
    cache['prob']=prob.tolist()


    if args.save :
        with open(os.path.join(args.save_dir,"sexual_prec_rec.txt"), "a") as f:
            f.write("\n")
            # f.write("image_model: {} saved_model: {} test acc: {:.3f}".format(args.model,os.path.splitext(os.path.split(args.path)[1])[0],accuracy))
            f.write("image_model: {} saved_model: {} test rec: {:.3f} test prec: {:.3f} F1 {:.3f}".format(args.model,os.path.splitext(os.path.split(args.path)[1])[0],recall,precision,F1))
        # cache_data=pd.DataFrame(cache)
        # print(cache_data.head())
        # cache_data.to_csv(os.path.join(args.save_dir,'sexual4_{}.csv'.format(os.path.splitext(os.path.split(args.path)[1])[0])),encoding='utf-8')

        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="ViT-B/32")
    parser.add_argument("--image_dir","-i",default="../SexualIntentDetection-master/Images")
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--save_dir",type=str,default="../SexualIntentDetection-master/result")
    parser.add_argument("--path","-p", type=str, default="../final_log/original_ViT-B32_0.001_16_500.pkl")
    parser.add_argument("--find","-f",type=bool,default=False)
    args = parser.parse_args()
    
    if args.find == False:
        if args.model.replace("/","") not in args.path :
            raise ValueError
        else :
            main(args,0.5)

    else :
        path='../final_log'
        for i in os.listdir(path) :
            args.path=os.path.join(path,i)
            j=os.path.splitext(i)[0]
            model=j.split('_')[1]
            if model in ['ViT-B16','ViT-B32','ViT-L14'] :
                args.model= model[:5]+'/'+model[5:]
                # print(args.model)
            else :
                args.model=model
            print(args)
            main(args,0.5)
