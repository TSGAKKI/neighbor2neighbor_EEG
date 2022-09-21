import numpy as np
import scipy.io as scio
from os import path
from scipy import signal
import math

def get_rms(records):   
    # return math.sqrt( sum([x ** 2 for x in records]) / len(records) ) 
    return np.sqrt( np.mean(records**2,axis=-1))
 
def read_psg(path_Extracted, sub_id, channels, resample=3000):
    psg = scio.loadmat(path.join(path_Extracted, 'subject%d.mat' % (sub_id)))
    psg_use = []
    for c in channels:
        psg_use.append(
            np.expand_dims(signal.resample(psg[c], resample, axis=-1), 1))# 原本psg是一个dict, 纬度是6000，应该是200Hz
    psg_use = np.concatenate(psg_use, axis=1)
    return psg_use

def read_noise(path_Extracted, sub_id, channels, resample=3000):
    psg = scio.loadmat(path.join(path_Extracted, 'subject%d.mat' % (sub_id)))
    psg_use = []
    for c in channels:
        psg_use.append(
            np.expand_dims(signal.resample(psg[c], resample, axis=-1), 1))# 原本psg是一个dict, 纬度是6000，应该是200Hz
    psg_use = np.concatenate(psg_use, axis=1)
    return psg_use

def read_label(path_RawData, sub_id, ignore=30):
    label = []
    with open(path.join(path_RawData, '%d/%d_1.txt' % (sub_id, sub_id))) as f:
        s = f.readline()
        while True:
            a = s.replace('\n', '')
            label.append(int(a))
            s = f.readline()
            if s == '' or s == '\n':
                break
    return np.array(label[:-ignore])

def contaminate(clean, artifact_ori, noise_position, times = 1):

    # SNR_val_dB = np.linspace(-7.0, 2.0, num=(10))
    SNR_val_dB = 0
    SNR_val = 10 ** (0.1 * (SNR_val_dB))
    artifact = np.zeros((artifact_ori.shape[0],1,artifact_ori.shape[-1]))
    for c in range(artifact_ori.shape[1]):
       artifact +=  artifact_ori[:,c:c+1,:]     

    noise_EEG = []
    for i in range(times): # 1-10 snr ratio
        noise_eeg_val = []
        for j in range(clean.shape[0]): # batch-level   
            eeg = clean[j]
            noise = artifact[j]
            
            if noise_position == 'anywhere':
                time_steps = artifact.shape[-1]
                noise_length = time_steps//3
                random_start_positon = np.random.randint(0,time_steps-1-noise_length)
                neeg = eeg

                noise = noise[:,random_start_positon:random_start_positon + noise_length]
                eeg = eeg[:,random_start_positon:random_start_positon + noise_length]
                coe = get_rms(eeg) / (get_rms(noise) * SNR_val)
                noise = noise * coe.reshape(-1,1)
                neeg[:,random_start_positon:random_start_positon + noise_length] = noise + neeg[:,random_start_positon:random_start_positon + noise_length]
                
            else:
                coe = get_rms(eeg) / (get_rms(noise) * SNR_val)
                noise = noise * coe.reshape(-1,1)
                neeg = noise + eeg
            
            noise_eeg_val.append(neeg)
        
        noise_EEG.extend(noise_eeg_val)
    noise_EEG = np.stack(noise_EEG,axis=0)
    return noise_EEG

'''
output:
    save to $path_output/ISRUC_S3.npz:
        Fold_data:  [k-fold] list, each element is [N,V,T]
        Fold_label: [k-fold] list, each element is [N,C]
        Fold_len:   [k-fold] list
'''


def get_isruc(data_folder, noise_position, noise_type ):

    fold_label = []
    fold_clean = []
    fold_contaminated = []
    fold_len = []
    
    path_RawData = path.join(data_folder,'ISRUC_S3','RawData')
    path_Extracted = path.join(data_folder,'ISRUC_S3','ExtractedChannels')

    clean_channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1']
    
    if noise_type == 'EOG':
        noise_channels = [ 'LOC_A2', 'ROC_A1' ]
        
    if noise_type == 'EMG':
        noise_channels = [ 'X1', ]
            
    for sub in range(1, 11):

        label = read_label(path_RawData, sub)

        psg_clean = read_psg(path_Extracted, sub, clean_channels)

        noise = read_noise(path_Extracted, sub, noise_channels)
        
        contaminated_signals = contaminate(psg_clean, noise, noise_position)
        
        assert len(label) == len(psg_clean)

        # in ISRUC, 0-Wake, 1-N1, 2-N2, 3-N3, 5-REM

        label[label==5] = 4
        fold_label.append(np.eye(5)[label])
        fold_clean.append(psg_clean)
        fold_contaminated.append(contaminated_signals)
        fold_len.append(len(label))
        
        print('Read subject', sub, psg_clean.shape)
    
    return  fold_clean, fold_contaminated, fold_len