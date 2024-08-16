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
    epoch = 400
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    

    dataset = load_dataset("csv", data_files={'train': 'train_ano.csv', 'val': 'val_ano.csv'}, cache_dir= './cache')
    train_dataloader= DataLoader(dataset['train'], shuffle=True, batch_size=args.batch, collate_fn=collate_fn)
    total_steps = len(train_dataloader)*epoch

    model = DementiaModel()
    model.to(device)
    optimizer = AdamW(model.parameters(),
    lr = args.lr, 
    eps = 1e-8)

    for param in model.parameters():
        param.requires_grad = True
    
    loss_values = []

    for epoch_i in range(epoch):
        print(" ")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epoch))
        print('Training...')
        t0 = time.time()
        total_loss = 0
        correct_train = 0
        total_train = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step % 50 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            input = batch['input_feature'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            output = model.forward(input)
            loss = criterion(output, labels.unsqueeze(1))
            predictions = torch.round(output).squeeze(1)
            correct_train += (predictions == labels).sum().item()
            total_loss += loss
            loss.backward()
            total_train += len(predictions)
            optimizer.step()
            del output, input, labels, loss
            # scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)

        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training Accuracy:", str(correct_train/total_train))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))



        #validation after each epoch
        val_dataloader = DataLoader(dataset['val'], batch_size=args.batch, collate_fn=collate_fn)
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
            f.write('lr: ')
            f.write(str(args.lr))
            f.write('\n')
            f.write(str(epoch_i))
            f.write('\n')
            f.write('Training Acc:')
            f.write(str(correct_train/total_train))
            f.write('\n')
            f.write('Avg Training Loss:')
            f.write(str(avg_train_loss))
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

        # if U_recall(predicted_label, true_label) >= 0.6237:
        #     checkpoint_name = args.save_checkpoint + '_' + str(epoch_i)
        #     torch.save({
        #     'epoch': str(epoch_i),
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     }, checkpoint_name)

    # save the model
    torch.save({
            'epoch':200,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, args.save_checkpoint)
    
    print("Training complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, help='path to model')
    parser.add_argument('save_checkpoint', type=str, help='filename to the resulted checkpoint')
    parser.add_argument('lr', type=float, help='learning rate')
    parser.add_argument('filename', type=str, help='filename to store the results')
    parser.add_argument('batch', type=int, help='batch size')
    args = parser.parse_args()

    train(args)


        