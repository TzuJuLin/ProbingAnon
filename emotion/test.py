
import speechbrain as sb
from speechbrain.inference.interfaces import Pretrained, foreign_class
from datasets import load_dataset
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import soundfile as sf
import time
import datetime
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
from torchmetrics.classification import MulticlassF1Score, Recall, Precision, ConfusionMatrix
import pandas as pd

def preprocess_function(data):
    inputs = []
    for file in data['file']:
        signal, _ = sf.read('/mount/arbeitsdaten/analysis/lintu/RAVDESS/'+file)
        # signal, _ = sf.read('/mount/arbeitsdaten/analysis/lintu/VoicePAT/results/anon_speech/ims_sttts_pc/ravdess/'+file)
        
        inputs.append(signal)
    data['input_values']=inputs
    return data


def collate_fn(batch):
    
    max_len = max(len(data['input_values']) for data in batch)

    padded_input_values = []
    labels = []
    for data in batch:
        input_values = data['input_values']
        label = data['emotion']
        
    # Pad input_values to the maximum length
        padded_input_values.append(input_values + [0] * (max_len - len(input_values)))
        labels.append(label)

    # Convert to PyTorch tensors
    padded_input_values = torch.tensor(padded_input_values)
    labels = torch.tensor(labels)
    
    return {'input_values': padded_input_values, 'label': labels}



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier", run_opts={"device":"cuda"})
criterion = nn.CrossEntropyLoss()
f1= MulticlassF1Score(num_classes=4, average=None)
recall = Recall(task="multiclass", average='none', num_classes=4)
U_recall = Recall(task="multiclass", average="macro", num_classes=4)
precision = Precision(task="multiclass", average='none', num_classes=4)
confmat = ConfusionMatrix(task="multiclass", num_classes=4)

def test(args):
    gpu_id = args.gpu
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    

    model=classifier
    optimizer = AdamW(model.parameters())
    checkpoint=torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(device)


    dataset = load_dataset("csv", data_files={'test':'test.csv'}, cache_dir= './cache')
    dataset = dataset.remove_columns(['sentence'])
    encoded_dataset_annon = dataset.map(preprocess_function, batched=True)
    encoded_dataset_annon = encoded_dataset_annon.remove_columns(['file'])

 


    #validation after each epoch
    val_dataloader = DataLoader(encoded_dataset_annon['test'], batch_size=16, collate_fn=collate_fn)
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    true_label = []
    predicted_label = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            # Move data to GPU
            input_values = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            true_label += batch['label']
            
            # Forward pass
            output = model.classify_batch(input_values)
            
            # Calculate loss
            loss = criterion(output, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output, dim = 1)
            predicted_label += predicted

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    true_label = torch.tensor(true_label, dtype=torch.int)
    predicted_label = torch.tensor(predicted_label, dtype=torch.int)

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
    
    df = pd.read_csv('new_test_label.csv')
    df['predicted_ano'] = predicted_label
    df.to_csv('new_test_label.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str, help='path to model')
    parser.add_argument('gpu', type=int, help='GPU id')
    parser.add_argument('filename', type=str, help='filename to store the results')
    args = parser.parse_args()

    test(args)