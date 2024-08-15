import torch
import code3.zihan.Transformer3D as Transformer3D

path = ["/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/ckpts/ziqi_Resnet10/P1_ep_2_iter599.pth",
        "/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/ckpts/ziqi_Resnet10/P2_ep_2_iter599.pth",
        "/Data3/cao/ZiHanCao/huawei_contest/code3/zihan/ckpts/ziqi_Resnet10/P3_ep_1_iter1299.pth"]
names = ["P1", "P2", "P3"]
weight_path = "code3/zihan/ckpts/ziqi_Resnet10"

for p, n in zip(path, names):
    print('loading model pos:', n)
    model = torch.load(p)
    state_dict = model.state_dict()
    torch.save(state_dict, f"{weight_path}/{n}_state_dict.pth")
    

    