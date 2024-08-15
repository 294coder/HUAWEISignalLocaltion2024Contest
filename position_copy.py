import torch 
import numpy as np
from data_processing.data_processing import read_slice_of_file
import os
import matplotlib.pyplot as plt
slice_num = 20000
# def main():
    
#     model = MLP()
#     model.eval()
#     batch_size = 100
#     state_dict = torch.load(r'/Data3/cao/ZiHanCao/huawei_contest/ckpt/model_12.pth',map_location=torch.device('cpu'))
#     model.load_state_dict(state_dict)
#     #model.eval()
#     #model.train(True)
#     data_dir = r'/Data3/cao/ZiHanCao/huawei_contest/data_new/train_data'
#     path_list = os.listdir(data_dir)
#     #path_list.sort(key = lambda x:int(x.split('.')[0]))
#     data_files = [os.path.join(data_dir,f) for f in path_list]
#     data_list = []
#     for i,file in enumerate(data_files):
#         rowdata = np.load(file)
#         data_list.append(rowdata)
#     anchor_pos = np.zeros((slice_num,2))
#     for i in range(int(slice_num/batch_size)):
#         print('pred batch:'+str(i))
#         data = torch.Tensor(data_list[i*batch_size:(i+1)*batch_size])
#         anchor_pos[i*batch_size:(i+1)*batch_size,:] = model(data).detach().numpy()
#         print(data.shape)
#     np.save(r'rd_matrix.npy',anchor_pos)

def poscompare():
    pred = read_slice_of_file(r'/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/results/rotated5/Round3OutputPos3.txt',0,slice_num)
    rd_pos = np.loadtxt(pred)[:,:]
    data_len = len(rd_pos)
    gt_pos = np.zeros((data_len,2))
    slice_sam_num = batch_size = 100
    # for idx in range(int(data_len/slice_sam_num)):
    #     truth_line1  = read_slice_of_file(anchor_path,idx*slice_sam_num,(idx+1)*slice_sam_num)
    #     truth_value1 = np.loadtxt(truth_line1)
    #     gt_pos[idx*slice_sam_num:(idx+1)*slice_sam_num] = truth_value1[:,1:3].reshape(slice_sam_num,2)
    # np.save(r'rd_matrix_gt.npy',gt_pos)
    plot_color(rd_pos,rd_pos)
    
def plot_color(preds,gt):
    center_point = np.zeros(2,dtype = np.float32)
    center_point[0] = 0.5*(np.min(gt[:,0],axis = 0)+np.max(gt[:,0],axis=0))
    center_point[1] = 0.5*(np.min(gt[:,1],axis = 0)+np.max(gt[:,1],axis=0))
    normalize = lambda in_data:(in_data-np.min(in_data))/(np.max(in_data)-np.min(in_data))
    rgb = np.zeros((gt.shape[0],3))
    rgb[:,0] = 0.8 # 0.3 #0.7*normalize(gt[:,0])
    rgb[:,1] = 0.6 #0.8 # *normalize(np.square(np.linalg.norm(gt-center_point,axis=1)))
    rgb[:,2] = 0.1 # 0.7 # *normalize(gt[:,1])
    
    plt.figure(figsize=(6,6))
    plt.scatter(preds[:,0],preds[:,1],c = rgb, s=12)
    plt.savefig('/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/results/rotated5/p3.png',dpi = 600)
    #plt.show()

if __name__ =='__main__':
    #anchor_path = r"/Data3/cao/ZiHanCao/datasets/huawei/Round1/Round1InputPos3.txt"
    #rd_pos_path = r'rd_matrix.npy'
    #main()
    poscompare()