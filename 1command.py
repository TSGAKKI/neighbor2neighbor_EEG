import os
import numpy as np
import time
import argparse

device = 3
seed = 1
data_dir = '/data/JinXiyuan/pyspace/data/'
# save_dir = 'result'

position = 'full'
noise_type = 'EOG'
num_epochs = 100

subject_independent = 1
debug = 0

com_code = f'python main.py --debug {debug} --rand_seed {seed} --cuda {device} --data_dir {data_dir} --subject_independent {subject_independent} --num_epochs {num_epochs} --position {position} --noise_type {noise_type}'  
 
start_time = time.asctime(time.localtime(time.time()))
os.system(com_code)
print('\nstart  at',start_time)
print('finish at',time.asctime(time.localtime(time.time())))