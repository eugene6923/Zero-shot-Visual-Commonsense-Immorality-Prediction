import glob
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append('../.')
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

    if args.model in ['ViT-L/14'] :
        classification2=nn.DataParallel(ClassificationHead2(dropout).to(device),device_ids=[0,1,2,3])
        classification =torch.load(args.path)
        classification2.load_state_dict(classification)

    elif args.model in ['ViT-B/32', 'ViT-B/16']:
        classification2=nn.DataParallel(ClassificationHead3(dropout).to(device),device_ids=[0,1,2,3])
        classification =torch.load(args.path)
        classification2.load_state_dict(classification)  

    device2 = torch.device("cuda:2")
    model,preprocess2 = clip.load(args.model, device2)

    image_features=[]

    y = []
    cache={}
    files = []
    files = []
    contents=[]
    classes=[]
    i=0


    if args.save:
            with open(os.path.join(args.save_dir,"sexual.txt"), "a") as f:
                f.write("\n")

    for content in os.listdir(args.image_dir) :
        i2=0
        print(content)
        for j in os.listdir(os.path.join(args.image_dir,content,"IMAGES")) :
            try :
                if content in ["drawings","neutral"] and os.path.splitext(j)[1] in [".jpg",".png",".jpeg"]:
                    try : 
                        img_try = Image.open(os.path.join(args.image_dir,content,"IMAGES",j))
                        image_input = preprocess2(img_try).unsqueeze(0).to(device2)

                        with torch.no_grad():
                            image_features.append(torch.tensor(model.encode_image(image_input,device=2)).squeeze(dim=1))
                        
                        y.append(0)
                        classes.append(content)
                        files.append(os.path.join(args.image_dir,content,"IMAGES",j))
                        contents.append(j)
                        i2+=1
                    except TypeError :
                        pass
                elif content in ["sexy","porn","hentai"] and os.path.splitext(j)[1] in [".jpg",".png",".jpeg"]:
                    try : 
                        img_try = Image.open(os.path.join(args.image_dir,content,"IMAGES",j))
                        image_input = preprocess2(img_try).unsqueeze(0).to(device2)

                        with torch.no_grad():
                            image_features.append(model.encode_image(image_input))
                        y.append(1)
                        classes.append(content)
                        files.append(os.path.join(args.image_dir,content,"IMAGES",j))
                        contents.append(j)  
                        i2+=1
                    except TypeError :
                        pass
                else :
                    pass
            
            except TypeError :
                i+=1
                print(content)
                pass
        print(content, i2)

    print("There is {} files that cannot predict".format(i))

    y = np.array(y)
    cache['label']=y
    cache['image_features']=[]
    for i,features in enumerate(image_features) :
        cache['image_features'].append(torch.tensor(features,device=device2).cpu().numpy())

    x=torch.tensor([item.cpu().detach().numpy() for item in image_features]).cuda() 
    y=torch.tensor(y)
    data=TensorDataset(x,y)
    test_dataloader = DataLoader(data, batch_size=64, shuffle=False)
    accuracy=evaluate(classification2,test_dataloader)
    precision, recall,F1=evaluate2(classification2,test_dataloader)
    prob=get_probs(classification2,test_dataloader)
    cache['prob']=prob.tolist()

    if args.save :
        with open(os.path.join(args.save_dir,"nsfw_prec_recall.txt"), "a") as f:
            f.write("\n")
            f.write("image_model: {} saved_model: {} test acc: {:.3f}".format(args.model,os.path.splitext(os.path.split(args.path)[1])[0],accuracy))
            f.write("image_model: {} saved_model: {} test rec: {:.3f} test prec: {:.3f} F1 {:.3f}".format(args.model,os.path.splitext(os.path.split(args.path)[1])[0],recall,precision,F1))
        # cache_data=pd.DataFrame(cache)
        # print(cache_data.head())
        # cache_data.to_csv(os.path.join(args.save_dir,'sexual3_{}.csv'.format(os.path.splitext(os.path.split(args.path)[1])[0])),encoding='utf-8')

        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="ViT-B/32")
    parser.add_argument("--image_dir","-i",default="../nsfw_data_scrapper/image")
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--save_dir",type=str,default="../nsfw_data_scraper/results")
    parser.add_argument("--path","-p", type=str, default="../final_log/original_ViT-B32_0.002_64_100.pkl")
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
            main(args,0.5)
