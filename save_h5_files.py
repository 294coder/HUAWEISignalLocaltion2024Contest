import numpy as np
import glob
import h5py
import os
from tqdm import tqdm
import torch
from dataclasses import dataclass


@dataclass
class Args:
    base_path = '/Data3/cao/ZiHanCao/huawei_contest/data/Round2Pos1/test_gt_data_64_64/'
    interpolate = True
    h5_shapes = [64, 64, 64, 4]
    
def main():
    args = Args()
    file_list = glob.glob(args.base_path + '/*.npy')
    # sort
    file_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    print('file_list:', file_list)
    print('\n\n =================================================================== \n\n')
    input('check file_list whether is in order and press any key to continue...')
    
    args.parent_name = os.path.dirname(args.base_path)
    args.h5_filename = os.path.basename(args.base_path)
    save_path = args.parent_name + '/' + args.h5_filename + '.h5'
    args.save_path = save_path
    
    print('save_path:', save_path)
    h5_data = h5py.File(save_path, 'w')

    print('we have {} files'.format(len(file_list)))
    h5_data.create_dataset('data', (len(file_list), *args.h5_shapes), dtype=np.float32, chunks=(1, 64, 64, 64, 4))

    tbar = tqdm(enumerate(file_list), total=len(file_list))
    for i, p in tbar:
        data = np.load(p)
        print(data.shape)
        data = data.astype(np.float32)  # [256, 128, 128, 4]
        
        # interpolate
        if args.interpolate:
            th_data = torch.tensor(data).type(torch.float32).unsqueeze(0)  # [1, 256, 128, 128, 4]
            th_data = th_data.permute(0, -1, 1, 2, 3)
            th_data = torch.nn.functional.interpolate(th_data, size=args.h5_shapes[:3],
                                                      mode='trilinear', align_corners=False)
            data = th_data[0].permute(1, 2, 3, 0).numpy()
        
        h5_data['data'][i] = data
        tbar.set_description('save [{}/{}] data'.format(i, len(file_list)))
        
    h5_data.close()
    print('save h5 file done.')


if __name__ == '__main__':
    print('TASK: save h5 files')
    
    main()
