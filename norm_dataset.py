from pathlib import Path
import numpy as np
import h5py
from rich.console import Console
from tqdm import trange

console = Console(log_path=False)

MEAN = np.array([-5.091097, -5.1093035, -5.0822234, -5.0828133])[None, None, None]
STD = np.array([5.861487, 5.879711, 5.8663826, 5.875907])[None, None, None]

# load h5 file
pos_n = 1
small_file_n = 2000
precomp_index = 0
console.log('loading test data of pos_n:', pos_n)
file = h5py.File(f"/Data3/cao/ZiHanCao/huawei_contest/data/Round3Pos{pos_n}/test_gt_64.h5")['data']
shape = file.shape


# load data
console.log('loading data...')
for i in range(precomp_index * small_file_n, shape[0], small_file_n):
    file_name = str(i // small_file_n).zfill(2)
    file_path = f"/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/h5file/R3P{pos_n}/{file_name}_normed.h5"
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    normed_file = h5py.File(file_path, 'w')
    normed_file.create_dataset('data', (small_file_n, 4, 64, 64, 64), dtype=np.float32, chunks=(1, 4, 64, 64, 64))
    console.log(f'create {file_path} file')
    normed_data = normed_file['data']
    
    tbar = trange(small_file_n)
    for j in tbar:
        data = file[i+j]
        data = (data - MEAN) / STD
        # [64, 64, 64, 4] -> [4, 64, 64, 64]
        normed_data[j] = np.transpose(data, (3, 0, 1, 2))
        tbar.set_description(f'from {i+j} to {j} of {file_name} h5 file.')
    
    normed_file.close()
console.log(f'normalize the dataset {file_name} of pos_{pos_n} done!')
                                                          