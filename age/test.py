
#set up cache path
import os
os.environ['TRANSFORMERS_CACHE'] = './cache/'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor,TrainingArguments, Trainer, AdamW, get_linear_schedule_with_warmup, Wav2Vec2Model
import torch
import torch.nn as nn
import pandas as pd
import soundfile as sf
import time
import datetime
from torch.utils.data import DataLoader
import numpy as np
from torch import tensor
from torchmetrics.classification import MulticlassF1Score, Recall, Precision, ConfusionMatrix
import argparse


sampling_rate = 16000

def preprocess_function_anno(data_split):
    processor = Wav2Vec2Processor.from_pretrained("audeering/wav2vec2-large-robust-6-ft-age-gender")
    audio_array = []
    for file in data_split['path']:
        signal, wr = sf.read('/mount/arbeitsdaten/analysis/lintu/VoicePAT/results/anon_speech/ims_sttts_pc/cv_test/' + file[16:-4] +'.wav')
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


#load dataset
dataset = load_dataset("csv", data_files={'val':'test.csv'}, cache_dir= './cache')
dataset = dataset.remove_columns(['index','sentence', 'gender', 'speaker_id', 'age', 'num_age'])

encoded_dataset_annon = dataset.map(preprocess_function_anno, batched=True)
encoded_dataset_annon = encoded_dataset_annon.remove_columns(['path', 'attention_mask'])

# train_dataloader_anno = DataLoader(encoded_dataset_annon['train'], shuffle=True, batch_size=2, collate_fn=collate_fn)
# val_dataloader_anno = DataLoader(encoded_dataset_annon['val'], batch_size=2, collate_fn=collate_fn)
test_dataloader_anno = DataLoader(encoded_dataset_annon['val'], batch_size=4, collate_fn=collate_fn)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Load the pretrained model and processor
model_name = "audeering/wav2vec2-large-robust-6-ft-age-gender"
model = Wav2Vec2Model.from_pretrained(model_name)

# Access the Wav2Vec encoder layers
wav2vec_encoder = model
#loss function
criterion = nn.CrossEntropyLoss()

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
    

f1= MulticlassF1Score(num_classes=3, average=None)
recall = Recall(task="multiclass", average='none', num_classes=3)
U_recall = Recall(task="multiclass", average="macro", num_classes=3)
precision = Precision(task="multiclass", average='none', num_classes=3)
confmat = ConfusionMatrix(task="multiclass", num_classes=3)

def test(args):

    model = AgeModel()
    optimizer = AdamW(model.parameters())

    gpu_id = args.gpu
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    true_label = []
    predicted_label = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader_anno):
            # Move data to GPU
            input_values = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            true_label += batch['label']
            
            # Forward pass
            logits_age = model(input_values)
            
            # Calculate loss
            loss = criterion(logits_age, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits_age, 1)
            predicted_label += predicted

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    true_label = torch.tensor(true_label, dtype=torch.int)
    predicted_label = torch.tensor(predicted_label, dtype=torch.int)
    
    with open (args.filename, 'a') as f:
        f.write("loss: ")
        f.write(str(val_loss / len(test_dataloader_anno)))
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
    
    df = pd.read_csv('ano_all.csv')
    df['predicted'] = predicted_label
    df.to_csv('processed_results_label.csv', index=False)

torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str, help='path to model')
    parser.add_argument('filename', type=str, help='filename to store the results')
    parser.add_argument('gpu', type=int, help='GPU id')
    args = parser.parse_args()

    test(args)



        