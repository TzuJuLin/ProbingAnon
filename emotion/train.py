
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

def preprocess_function(data):
    inputs = []
    for file in data['file']:
        #change path to the location of the audio files (either original or anonymized path)
        signal, wr = sf.read('/mount/arbeitsdaten/analysis/lintu/VoicePAT/results/anon_speech/ims_sttts_pc/ravdess/'+file)
        inputs.append(signal)
    data['input_values']=inputs
    return data


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


def train(args):
    epoch = 30
    gpu_id = args.gpu
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    if args.checkpoint_path:
        model=classifier
        optimizer = AdamW(model.parameters(),
            lr = args.lr, 
            eps = 1e-8 
        )
        checkpoint=torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.to(device)


    else:
        model = classifier
        model = model.to(device) 
        a = list(model.parameters())[0]
        optimizer = AdamW(model.parameters(),
            lr = args.lr, 
            eps = 1e-8 
        )
        
        
    for param in model.parameters():
        param.requires_grad = True

    dataset = load_dataset("csv", data_files={'train': 'train.csv', 'val':'val.csv'}, cache_dir= './cache')
    dataset = dataset.remove_columns(['sentence'])
    encoded_dataset_annon = dataset.map(preprocess_function, batched=True)
    encoded_dataset_annon = encoded_dataset_annon.remove_columns(['file'])
    train_dataloader= DataLoader(encoded_dataset_annon['train'], shuffle=True, batch_size=args.batch, collate_fn=collate_fn)
    loss_values = []
    real_train = []
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
            real_train.append(batch['label'])
            input = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            output = model.classify_batch(input)
            _, output_label_train = torch.max(output, dim=1)
            loss = criterion(output, labels)
            total_loss += loss
            loss.backward()
            total_train += labels.size(0)
            correct_train += (output_label_train == labels).sum().item()
            optimizer.step()
        
            del output, input, labels, loss

        

        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)

        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training Accuracy:", str(correct_train/total_train))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))



        #validation after each epoch
        val_dataloader = DataLoader(encoded_dataset_annon['val'], batch_size=args.batch, collate_fn=collate_fn)
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
            f.write('lr: ')
            f.write(str(args.lr))
            f.write('\n')
            f.write(str(epoch_i))
            f.write('\n')
            f.write('Training Loss:')
            f.write(str(avg_train_loss))
            f.write('\n')
            f.write('Training Acc:')
            f.write(str(correct_train/total_train))
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
        
        # Was used to save models that were good enough
        # if correct / total >= 0.4583:
        #     checkpoint_name = args.save_checkpoint + '_' + str(epoch_i)
        #     torch.save({
        #     'epoch': str(epoch_i),
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     }, checkpoint_name)


    # save the model
    torch.save({
            'epoch':30,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, args.save_checkpoint)
    
    print("Training complete!")

  

        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, help='path to model')
    parser.add_argument('save_checkpoint', type=str, help='filename to the resulted checkpoint')
    parser.add_argument('gpu', type=int, help='GPU id')
    parser.add_argument('lr', type=float, help='learning rate')
    parser.add_argument('filename', type=str, help='filename to store the results')
    parser.add_argument('batch', type=int)
    args = parser.parse_args()

    train(args)