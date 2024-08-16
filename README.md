# 华为无线通信菁英赛
队名：winkwink

## 环境配置
请运行下面命令安装环境：

```shell
pip install -r requirementList.txt
```

> 数据处理代码依赖于C代码，请**务必使用**Python 3.9或者3.12，也许支持3.7，3.8和3.10。

## 测试

如需测试，请在下面链接下载*预训练模型*和*预处理数据*，将他们按照如下路径放置：
- 预训练模型链接：https://pan.baidu.com/s/1GIINu_-FQzP3X1QyThX5Ow?pwd=dph4 提取码: dph4 
- 预处理数据连接：https://pan.baidu.com/s/1ofO1cGxwKfy6YG2fxAzHCw?pwd=2585 提取码: 2585 


```
预训练模型位置：
<root>/ckpts
├── R3P1.pth
├── R3P2.pth
└── R3P3.pth

预处理数据位置：
<root>/h5file
├── R3P1
│   ├── 00_normed.h5
│   ├── 01_normed.h5
│   ├── 02_normed.h5
│   ├── 03_normed.h5
│   └── ...
├── R3P2
│   ├── 00_normed.h5
│   ├── 01_normed.h5
│   ├── 02_normed.h5
│   ├── 03_normed.h5
│   └── ...
├── R3P3
│   ├── 00_normed.h5
│   ├── 01_normed.h5
│   ├── 02_normed.h5
│   ├── 03_normed.h5
│   └── ...   
```

确保路径正确后，运行测试代码：
```sh
python test_gt_transformer.py --pos_n <POS_N> --loading_strategy <LS>
```
其中，`<POS_N>`是三个场景，分别是`(1,2,3)`，`<LS>`是数据loading策略，可以是`(ALL, PART, ONFLY)`。
为了提高测试速度，请依据不同的机器配置选择不同的预处理数据loading方式：
1. RAM大于64G，我们推荐将`<LS>`设置为`ALL`；
2. RAM大于32G，我们推荐将`<LS>`设置为`PART`；
3. RAM小于32G，我们推荐将`<LS>`设置为`ONFLY`；

> 需要注意的是：
> (1) 使用`ONFLY`进行测试时，由于DataLoader的限制，测试速度很慢。实际上，在决赛中，我们使用256G的服务器进行测试，3个场景仅需5分钟；
> (2) 在使用`PART`进行测试时，由于我们的模型消耗的显存不大，可以launch 3个进程同时进行测试（实际上，每个场景的显存消耗仅2G)。

测试完成后，您会在`results/`目录下找到结果，命名为`Round3OutputPos<POS_N>.txt`。

> 为了方便测试，我们将转换扇区的后处理代码，从Numba迁移到Pytorch，实际在CUDA计算时与提交的结果有0.001~0.01的误差，这取决于NVCC、CUDA版本以及显卡配置。


## 数据处理
见`data_processing/__init__.py`中的`data_transforming`的函数签名。

## 模型训练

我们将模型的训练分为预训练和微调：
1. 在Round2和Round3的数据上进行训练后；
2. 再在单个场景上进行微调。

训练代码见`train_transformer3d_p1.py`。

这里以pos1为例，首先进行预训练：
```sh
python train_transformer3d_p1.py --data_dir1 <D1> --data_dir2 <D2> --data_dir3 <D3> --data_dir4 <D4> --anchor_path <AP1> --anchor_path2 <AP2> --test_data_dir <TD> --test_anchor_path <TAP> --prefix_weight_name "TransformerP1"
```
其中，`data_dir1`是Round3单个场景的h5训练数据，`data_dir{2-4}`是Round2的预训练h5数据，`anchor_path`是Round3单个场景的gt，`anchor_path2`是Round2的gt，`test_data_dir`是Round3单个场景的h5测试数据，`test_anchor_path`是Round3单个场景测试gt。


预训练完成后，模型文件将会存在`ckpt/TransformerP1`中，再对单个场景进行微调：
```sh
python train_transformer3d_p1.py --data_dir1 <D1> --anchor_path <AP1> --test_data_dir <TD> --test_anchor_path --pretrained-weight <PW> --finetune --prefix_weight_name "TransformerP1"
```

训练完成后，模型文件将会存在`ckpt/finetuneTransformerP1`中。