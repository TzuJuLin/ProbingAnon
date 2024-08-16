#set up cache path
import os
os.environ['TRANSFORMERS_CACHE'] = './cache/'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from datasets import load_dataset
from transformers import Wav2Vec2Processor, AdamW, get_linear_schedule_with_warmup,Wav2Vec2Model
import torch
import pandas as pd
import soundfile as sf
from tqdm import tqdm, trange
import time
import datetime
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import argparse


sampling_rate = 16000

def preprocess_function_anno(data_split):
    processor = Wav2Vec2Processor.from_pretrained("audeering/wav2vec2-large-robust-6-ft-age-gender")
    audio_array = []
    for file in data_split['path']:
        signal, wr = sf.read('/mount/arbeitsdaten/analysis/lintu/VoicePAT/results/anon_speech/ims_sttts_pc/cv_train/' + file[16:-4] + '.wav')
        audio_array.append(signal)
    inputs = processor(audio_array, sampling_rate=sampling_rate)
    return inputs


def collate_fn(batch):
    
    max_len = max(len(data['input_values']) for data in batch)

    padded_input_values = []
    labels = []
    for data in batch:
        input_values = data['input_values']
        label = data['mapped_age']
        
    # Pad input_values to the maximum length
        padded_input_values.append(input_values + [0] * (max_len - len(input_values)))
        labels.append(label)

    # Convert to PyTorch tensors
    padded_input_values = torch.tensor(padded_input_values)
    labels = torch.tensor(labels)
    
    return {'input_values': padded_input_values, 'label': labels}




# Load the pretrained model and processor
model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
processor = Wav2Vec2Processor.from_pretrained(model_name)
wav2vec_encoder = Wav2Vec2Model.from_pretrained(model_name)



class ModelHead(nn.Module):
    r"""Classification head."""

    def __init__(self, num_labels):

        super().__init__()

        self.dense = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(1024, num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x

class AgeModel(nn.Module):
    

    def __init__(self):
        super().__init__()
        self.wav2vec = wav2vec_encoder
        self.age = ModelHead(3)
    
    def forward(self, input_values):
        outputs = self.wav2vec(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = torch.softmax(self.age(hidden_states), dim=1)
        
        return logits_age

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

#loss function
criterion = nn.CrossEntropyLoss()

#or should I use a trainer instead?
def train(args):
    epoch=1
    gpu_id = args.gpu
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    

    if args.checkpoint_path:
        model = AgeModel()
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer = AdamW(model.parameters(),
            lr = args.lr, 
            eps = 1e-8 
        )
        
    
    else:
        model = AgeModel()
        model.to(device)
        optimizer = AdamW(model.parameters(),
            lr = args.lr, 
            eps = 1e-8 
        )

        
    
    dataset = load_dataset("csv", data_files={'train': args.training_data}, cache_dir= './cache')
    dataset = dataset.remove_columns(['index','sentence', 'gender', 'speaker_id', 'age', 'num_age'])
    encoded_dataset_annon = dataset.map(preprocess_function_anno, batched=True)
    encoded_dataset_annon = encoded_dataset_annon.remove_columns(['path', 'attention_mask'])
    train_dataloader= DataLoader(encoded_dataset_annon['train'], shuffle=True, batch_size=4, collate_fn=collate_fn)
    total_steps = len(train_dataloader)*epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps=0,
                                              num_training_steps = total_steps)
    loss_values = []
    for epoch_i in range(epoch):
      print(" ")
      print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epoch))
      print('Training...')
      t0 = time.time()
      total_loss = 0
      model.train()
      for step, batch in enumerate(train_dataloader):
        if step % 100 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        input = batch['input_values'].to(device)
        labels = batch['label'].to(device)
        model.zero_grad()
        logits_age = model(input)
        loss = criterion(logits_age, labels)
        total_loss += loss
        loss.backward()
        del logits_age, input, labels
        optimizer.step()
        scheduler.step()
        del loss

    avg_train_loss = total_loss / len(train_dataloader)
    loss_values.append(avg_train_loss)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    #save the model
    torch.save({
            'epoch': 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, args.save_checkpoint)
    
    print("Training complete!")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, help='path to model')
    parser.add_argument('save_checkpoint', type=str, help='filename to the resulted checkpoint')
    parser.add_argument('training_data', type=str, help='path to the training data')
    parser.add_argument('gpu', type=int, help='GPU id')
    parser.add_argument('--lr', type=float, help='learning rate')
    args = parser.parse_args()

    train(args)
        
    