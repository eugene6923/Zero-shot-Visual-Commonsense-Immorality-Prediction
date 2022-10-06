import pandas as pd
import os
import torch
import clip
from torch.utils.data import TensorDataset
import warnings
warnings.filterwarnings("ignore")
from torch import nn


def load_cm_sentences(data_dir, split="train"):
    if "long" in split:
        path = os.path.join(data_dir, "cm_{}.csv".format(split.split("long_")[1]))
        df = pd.read_csv(path)
        df = df[df["is_short"] == False]
    elif "short" in split:
        path = os.path.join(data_dir, "cm_{}.csv".format(split.split("short_")[1]))
        df = pd.read_csv(path)
        df = df[df["is_short"] == True]
    else:
        path = os.path.join(data_dir, "cm_{}.csv".format(split))
        df = pd.read_csv(path)

    if split == "ambig":
        labels = [-1 for _ in range(df.shape[0])]
        sentences = [df.iloc[i, 0] for i in range(df.shape[0])]
        
    else:
        labels = [df.iloc[i, 0] for i in range(df.shape[0])]
        sentences = [df.iloc[i, 1] for i in range(df.shape[0])]

    return sentences, labels

def load_optim(model,args) :

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.epsilon)

    return optimizer

def load_data(device,args,split) :

    class TextCLIP(nn.Module):
        def __init__(self, model) :
            super(TextCLIP, self).__init__()
            self.model = model
            
        def forward(self,text):
            return self.model.encode_text(text)
    
    clip_model=clip.load(args.clip,device=device)[0]

    model_text=TextCLIP(clip_model)
    data,data_label = load_cm_sentences(args.data_dir, split)
    data=clip.tokenize(data,args.max_length,truncate=True).to(device)

    with torch.no_grad():
        model_text=torch.nn.DataParallel(model_text,device_ids=[i for i in range(args.ngpus)])
        data=model_text(data)

    data=TensorDataset(torch.tensor(data),torch.tensor(data_label))
    return data

