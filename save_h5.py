import h5py
from tqdm import trange

pos_n = 3
test_dir = f"/Data3/cao/ZiHanCao/huawei_contest/data/Round3Pos{pos_n}/test_gt_64.h5"
file = h5py.File(test_dir, 'r')
print(file['data'].chunks)
if file['data'].chunks is None:
    file2 = h5py.File(f'/Data4/exps/dataset/Round3Pos{pos_n}_test_gt_64.h5', 'w')

    file2.create_dataset('data', shape=(20000, 64, 64, 64, 4), compression='gzip', chunks=(1, 64, 64, 64, 4))
    try:
        for i in trange(20000):
            file2['data'][i] = file['data'][i]
    except:
        file.close()
        file2.close()
