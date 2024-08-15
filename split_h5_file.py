import os
from pathlib import Path
import numpy as np
import h5py
from tqdm import trange


def split_h5_file_to_small_files(path):
    parent_dir = os.path.dirname(path)
    name = Path(path).stem
    file = h5py.File(path, 'r')
    data = file['data']
    
    small_n = 2000
    chunks = (1, 4, 64, 64, 64)
    for i in range(0, data.shape[0], small_n):
        small_file_n = str(i).zfill(2)
        small_file_name = f'{parent_dir}/{name}/{small_file_n}.h5'
        Path(small_file_name).parent.mkdir(parents=True, exist_ok=True)
        
        print(f'saving {small_file_name}...')
        print(f'from {i} to {i+small_n}')
        
        small_file = h5py.File(small_file_name, 'w')
        small_file.create_dataset('data', (small_n, *chunks[1:]), dtype=np.float32, chunks=chunks)
        tbar = trange(small_n)
        for j in tbar:
            small_file['data'][j] = data[i+j]
            tbar.set_description(f'from {i+j} to {j} of {small_file_n} h5 file.')
        small_file.close()
        print('done.')
        
split_h5_file_to_small_files('/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/h5file/R3P1.h5')
    