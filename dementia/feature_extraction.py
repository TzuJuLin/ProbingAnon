import soundfile as sf
import pandas as pd
import librosa
import numpy as np
from scipy import signal
from spafe.features.gfcc import gfcc
import scipy.io.wavfile
import os
import numpy as np
from scipy.io import wavfile
import parselmouth 
from parselmouth.praat import call
from IPython.display import Audio

def formants_praat(fs, x):
        f0min, f0max  = 75, 300
        sound = parselmouth.Sound(x, sampling_frequency=fs) # read the sound
        pitch = sound.to_pitch()
        f0 = pitch.selected_array['frequency']
        formants = sound.to_formant_burg(time_step=0.025, maximum_formant=5500)
        
        f1_list, f2_list, f3_list, f4_list  = [], [], [], []
        for t in formants.ts():
            f1 = formants.get_value_at_time(1, t)
            f2 = formants.get_value_at_time(2, t)
            f3 = formants.get_value_at_time(3, t)
            f4 = formants.get_value_at_time(4, t)
            if np.isnan(f1): f1 = 0
            if np.isnan(f2): f2 = 0
            if np.isnan(f3): f3 = 0
            if np.isnan(f4): f4 = 0
            f1_list.append(f1)
            f2_list.append(f2)
            f3_list.append(f3)
            f4_list.append(f4)
        f1_mean = sum(f1_list)/len(f1_list)
        f2_mean = sum(f2_list)/len(f2_list)
        f3_mean = sum(f3_list)/len(f3_list)
        f4_mean = sum(f4_list)/len(f4_list)
            
        return f1_mean, f2_mean, f3_mean, f4_mean

def get_feature(filepath):
    data, sampling_rate = sf.read(filepath)
    fs, sig = scipy.io.wavfile.read(filepath)
    
    #gfcc
    gfccs  = np.mean(gfcc(data, fs=sampling_rate, num_ceps=14), axis=0)  
    
    #mfcc
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=14).T, axis=0)
    #f0
    f0, _ , _ = librosa.pyin(data, sr=sampling_rate, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0 =  f0[~np.isnan(f0)]
    if len(f0) == 0:
        f0_mean = 0
    else:
        f0_mean = np.mean(f0)

    #energy
    energy = np.mean(np.log(np.sum(data**2)))

    #formants
    f1, f2, f3, f4 = formants_praat(sampling_rate, data)


    
    #convert all features into a single vector
    feature = mfcc.tolist()
    feature.append(f0_mean)
    feature.append(energy)
    feature.append(f1)
    feature.append(f2)
    feature.append(f3)
    feature.append(f4)
    feature += gfccs.tolist()
    
        
    return feature



df = pd.read_csv('val_con1.csv')

input = []

for i, file in enumerate(df['file']):
    if df['dementia'][i] == 0:
        filepath = '/mount/arbeitsdaten/analysis/lintu/VoicePAT/results/anon_speech/ims_sttts_pc/pitts_control_con/' + file + '.wav'
        # filepath = '/mount/arbeitsdaten/analysis/lintu/pitts/control/' + file + '.wav'
    else:
        filepath = '/mount/arbeitsdaten/analysis/lintu/VoicePAT/results/anon_speech/ims_sttts_pc/pitts_dementia_con/' + file + '.wav'
        # filepath = '/mount/arbeitsdaten/analysis/lintu/pitts/dementia/' + file + '.wav'

    try:
        feature = get_feature(filepath)


    except sf.LibsndfileError:
        feature = [0]
    
    input.append(feature)


new_df = pd.DataFrame({'file':df['file'], 'dementia':df['dementia'], 'input':input})
new_df.to_csv('val_ano.csv', index=False)

