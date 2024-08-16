import soundfile as sf
import pandas as pd
import librosa
import numpy as np


def get_feature(filepath):
    data, sampling_rate = sf.read(filepath)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=21).T, axis=0)
    f0, _ , _ = librosa.pyin(data, sr=sampling_rate, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 =  f0[~np.isnan(f0)]
    if len(f0) == 0:
        f0_mean = 0
        f0_max = 0
        f0_min = 0
        f0_median = 0
        f0_std = 0
    else:
        f0_mean = np.mean(f0)
        f0_max = np.max(f0)
        f0_min = np.min(f0)
        f0_median = np.median(f0)
        f0_std = np.std(f0)
    return mfcc, f0_mean, f0_max, f0_min, f0_median, f0_std

#use the file where all audio names are available 
df = pd.read_csv('/mount/arbeitsdaten/analysis/lintu/CV/train/train.csv', encoding = 'unicode_escape')

mfcc_lst = []
f0_mean_lst = []
f0_max_lst =[]
f0_min_lst = []
f0_median_lst = []
f0_std_lst = []


#loop through the target folder
for file in df['path']:
    filepath = '/mount/arbeitsdaten/analysis/lintu/VoicePAT/results/anon_speech/ims_sttts_pc/cv_train/' + file[16:-4] + '.wav'
    
    try:
        mfcc, f0_mean, f0_max, f0_min, f0_median, f0_std = get_feature(filepath)
        f0_mean_lst.append(f0_mean)
        f0_max_lst.append(f0_max)
        f0_min_lst.append(f0_min)
        f0_median_lst.append(f0_median)
        f0_std_lst.append(f0_std)
        mfcc_lst.append(mfcc.tolist())
    except sf.LibsndfileError:
        f0_mean_lst.append(0)
        f0_max_lst.append(0)
        f0_min_lst.append(0)
        f0_median_lst.append(0)
        f0_std_lst.append(0)
        mfcc_lst.append([0])


df['f0_mean']=f0_mean_lst
df['f0_max'] = f0_max_lst
df['f0_min'] = f0_min_lst
df['f0_median'] = f0_median_lst
df['f0_std'] = f0_std_lst
df['mfcc'] = mfcc_lst


#change to desired path and filename
df.to_csv('CV_with_feature_train_anno.csv', index=False)