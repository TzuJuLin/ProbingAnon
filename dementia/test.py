import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['TRANSFORMERS_CACHE'] = './cache/'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from datasets import load_dataset
from transformers import AdamW
import torch
import pandas as pd
from tqdm import tqdm, trange
import time
import datetime
from torch.utils.data import DataLoader
import numpy as np
import argparse
from torchmetrics.classification import MulticlassF1Score, Recall, Precision, ConfusionMatrix

class DementiaModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(34,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,4)
        self.output = nn.Linear(4,1)
        self.droupout = nn.Dropout(p=0.5)
        self.final = nn.Sigmoid()
    
    def forward(self, features, *kwargs):

        x = features
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.droupout(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.output(x)
        x = self.final(x)

        return x

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def collate_fn(batch):
    
    input_features = []
    labels = []

    for data in batch:
        feature = data['input'].split(", ")
        feature[0] = feature[0][1:]
        feature[-1] = feature[-1][:-1]
        feature = [float(a) for a in feature]
        features = np.array(feature, dtype='f')
        input_features.append(features)
        labels.append(data['dementia'])
    input_features = torch.FloatTensor(input_features)
    labels = torch.FloatTensor(labels)
    return {'input_feature':input_features, 'labels':labels}


criterion = nn.BCELoss()
f1= MulticlassF1Score(num_classes=2, average=None)
recall = Recall(task="multiclass", average='none', num_classes=2)
U_recall = Recall(task="multiclass", average="macro", num_classes=2)
precision = Precision(task="multiclass", average='none', num_classes=2)
confmat = ConfusionMatrix(task="multiclass", num_classes=2)

def train(args):

    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    

    dataset = load_dataset("csv", data_files={'test':'val_ano.csv'}, cache_dir= './cache')


    model = DementiaModel()
    optimizer = AdamW(model.parameters())
    checkpoint=torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(device)


        #validation after each epoch
    val_dataloader = DataLoader(dataset['test'], batch_size=args.batch, collate_fn=collate_fn)
    model.eval()
    val_loss = 0.0
    total = 0
    true_label = []
    predicted_label = []
    correct = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            # Move data to GPU
            input_values = batch['input_feature'].to(device)
            labels = batch['labels'].to(device)
            true_label += [int(i.item()) for i in batch['labels']]
            
            # Forward pass
            output = model.forward(input_values)
            
            # Calculate loss
            loss = criterion(output, labels.unsqueeze(1))
            val_loss += loss.item()

            # Calculate accuracy
            

            predictions = torch.round(output)
            predictions = [int(i.item()) for i in predictions]
            predicted_label += predictions
    for i, l in enumerate(predicted_label):
        if true_label[i] == predicted_label[i]:
            correct+=1
    total = len(predicted_label)



    true_label = torch.tensor(true_label)
    predicted_label = torch.tensor(predicted_label)

    with open (args.filename, 'a') as f:
        f.write('\n')
        f.write("loss: ")
        f.write(str(val_loss / len(val_dataloader)))
        f.write('\n')
        f.write("accuracy:")
        f.write(str(correct / total))
        f.write('\n')
        f.write("f1 score: ")
        f.write(str(f1(predicted_label, true_label)))
        f.write('\n')
        f.write("Recall: ")
        f.write(str(recall(predicted_label, true_label)))
        f.write('\n')
        f.write("Unweighted Average Recall: ")
        f.write(str(U_recall(predicted_label, true_label)))
        f.write('\n')
        f.write("Precision:")
        f.write(str(precision(predicted_label, true_label)))
        f.write('\n')
        f.write(str(confmat(predicted_label, true_label)))

    df = pd.read_csv('val_predicted_ano.csv')
    df['predicted'] = predicted_label
    df.to_csv('val_predicted_ano.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str, help='path to model')
    parser.add_argument('filename', type=str, help='filename to store the results')
    parser.add_argument('batch', type=int, help='batch size')
    args = parser.parse_args()

    train(args)


        