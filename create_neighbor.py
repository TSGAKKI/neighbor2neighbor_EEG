import numpy as np
import matplotlib.pyplot as plt
import torch

def subSample_EEG_Bi(raw_eeg,musk):
    EEG_SIZE=np.shape(musk)
    subSample_EEG1=[]
    subSample_EEG2=[]


    musk_index=0
    rear = EEG_SIZE[1]-2
    for j in range(EEG_SIZE[0]):
        subSample_EEG1_tmp=[]
        subSample_EEG2_tmp=[]
        musk_index=0
        subSample_EEG1_tmp.extend(raw_eeg[j][1:3])
        subSample_EEG2_tmp.extend(raw_eeg[j][1:3])
        for i in range(2,EEG_SIZE[1]-2,2):
            if(musk[j][musk_index]<0.5):
                subSample_EEG1_tmp.extend(raw_eeg[j][i-2:i])
                subSample_EEG2_tmp.extend(raw_eeg[j][i+1:i+3])
            else:
                subSample_EEG2_tmp.extend(raw_eeg[j][i-2:i])
                subSample_EEG1_tmp.extend(raw_eeg[j][i+1:i+3])
            musk_index = musk_index+1
        
        if(subSample_EEG1_tmp.size()[1]!=EEG_SIZE[1]/2):
            subSample_EEG1_tmp.extend(raw_eeg[j][rear:])
            subSample_EEG2_tmp.extend(raw_eeg[j][rear:])
           
        subSample_EEG1.append(subSample_EEG1_tmp)   
        subSample_EEG2.append(subSample_EEG2_tmp)
    print(type(subSample_EEG1))
    print(np.shape(subSample_EEG1))     
    print(np.shape(subSample_EEG2))
    return subSample_EEG1,subSample_EEG2

def create_musk(size,block_num=1):
    musk = np.random.random((size[0],int(size[1]/block_num)))
    print("musk shape:",np.shape(musk))

    return musk


if __name__ == '__main__':
    raw_eeg = np.load('/data/Liulei/preprocess/DeepSeparator-main/data/EOGtrain_input.npy')
    musk=create_musk(np.shape(raw_eeg),2)
    subSample_EEG_Bi(raw_eeg,musk)
    