import numpy as np
import matplotlib.pyplot as plt
def subSample_EEG_Bi(raw_eeg):
    EEG_SIZE=np.shape(raw_eeg)
    subSample_EEG1=[]
    subSample_EEG2=[]

    musk = np.random.rand(int(EEG_SIZE[1]/2))
    musk_index=0
    for j in range(EEG_SIZE[0]):
        subSample_EEG1_tmp=[]
        subSample_EEG2_tmp=[]
        musk_index=0
        subSample_EEG1_tmp.extend(raw_eeg[j][1:3])
        subSample_EEG2_tmp.extend(raw_eeg[j][1:3])
        for i in range(2,EEG_SIZE[1]-2,2):
            if(musk[musk_index]<0.5):
                subSample_EEG1_tmp.extend(raw_eeg[j][i-2:i])
                subSample_EEG2_tmp.extend(raw_eeg[j][i+1:i+3])
            else:
                subSample_EEG2_tmp.extend(raw_eeg[j][i-2:i])
                subSample_EEG1_tmp.extend(raw_eeg[j][i+1:i+3])
            musk_index = musk_index+1
        subSample_EEG1.append(subSample_EEG1_tmp)   
        subSample_EEG2.append(subSample_EEG2_tmp)   
    print(np.shape(subSample_EEG1))     
    print(np.shape(subSample_EEG2))
    return subSample_EEG1,subSample_EEG2