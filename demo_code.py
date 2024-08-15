import os

import time

import numpy as np

import itertools

 

 

# 核心定位功能函数，由参赛者实现

def calcLoc(H, anch_pos, bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num):

    '''

    H: 所有点位的信道

    anch_pos:锚点用户ID及坐标

    bs_pos: 基站坐标

    tol_samp_num: 总点位数

    anch_samp_num: 锚点点位数

    port_num: SRS Port数（UE天线数）

    ant_num: 基站天线数

    sc_num: 信道子载波数

    '''

    #########以下代码，参赛者用自己代码替代################

    #########样例代码中直接返回全零数据作为估计结果##########

    loc_result = np.zeros([tol_samp_num, 2], 'float')

    return loc_result

 

# 读取配置文件函数

def read_cfg_file(file_path):

    with open(file_path, 'r') as file:

        lines = file.readlines()

        line_fmt = [line.rstrip('\n').split(' ') for line in lines]

    info = line_fmt

    bs_pos = list([float(info[0][0]), float(info[0][1]), float(info[0][2])])

    tol_samp_num = int(info[1][0])

    anch_samp_num = int(info[2][0])

    port_num = int(info[3][0])

    ant_num = int(info[4][0])

    sc_num = int(info[5][0])

    return bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num

 

# 读取锚点位置文件函数

def read_anch_file(file_path, anch_samp_num):

    anch_pos = []

    with open(file_path, 'r') as file:

        lines = file.readlines()

        line_fmt = [line.rstrip('\n').split(' ') for line in lines]

    for line in line_fmt:

        tmp = np.array([int(line[0]), float(line[1]), float(line[2])])

        if np.size(anch_pos) == 0:

            anch_pos = tmp

        else:

            anch_pos = np.vstack((anch_pos, tmp))

    return anch_pos

 

# 切片读取信道文件函数

def read_slice_of_file(file_path, start, end):

    with open(file_path, 'r') as file:

        slice_lines = list(itertools.islice(file, start, end))

    return slice_lines

 

if __name__ == "__main__":

    print("<<< Welcome to 2024 Wireless Algorithm Contest! >>>\n")

    ## 不同轮次的输入数据可放在不同文件夹中便于管理，这里用户可以自定义

    PathSet = {0: "./Test", 1: "./CompetitionData1", 2: "./CompetitionData2", 3: "./CompetitionData3"}

    PrefixSet = {0: "Round0", 1: "Round1", 2: "Round2", 3: "Round3"}

 

    Ridx = 1  # 设置比赛轮次索引，指明数据存放目录。0:Test; 1: 1st round; 2: 2nd round ...

    PathRaw = PathSet[Ridx]

    Prefix = PrefixSet[Ridx]

   

    # 查找文件夹中包含的所有比赛/测试数据文件，非本轮次数据请不要放在目标文件夹中

    files = os.listdir(PathRaw)

    names = []

    for f in sorted(files):

        if f.find('CfgData') != -1 and f.endswith('.txt'):

            names.append(f.split('CfgData')[-1].split('.txt')[0])

   

    ## 创建对象并处理

    for na in names:

        FileIdx = int(na)

        print('Processing Round ' + str(Ridx) + ' Case ' + str(na))

       

        # 读取配置文件 RoundYCfgDataX.txt

        print('Loading configuration data file')

        cfg_path = PathRaw + '/' + Prefix + 'CfgData' + na + '.txt'

        bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num = read_cfg_file(cfg_path)

               

        # 读取锚点位置文件 RoundYInputPosX.txt

        print('Loading input position file')

        anch_pos_path = PathRaw + '/' + Prefix + 'InputPos' + na + '.txt'

        anch_pos = read_anch_file(anch_pos_path, anch_samp_num)

 

        # 读取信道文件 RoundYInputDataX.txt

        slice_samp_num = 1000  # 每个切片读取的数量

        slice_num = int(tol_samp_num / slice_samp_num)  # 切片数量

        csi_path = PathRaw + '/' + Prefix + 'InputData' + na + '.txt'

        H = []

        for slice_idx in range(2): # range(slice_num): # 分切片循环读取信道数据

            print('Loading input CSI data of slice ' + str(slice_idx))

            slice_lines = read_slice_of_file(csi_path, slice_idx * slice_samp_num, (slice_idx + 1) * slice_samp_num)

            Htmp = np.loadtxt(slice_lines)

            Htmp = np.reshape(Htmp, (slice_samp_num, 2, sc_num, ant_num, port_num))

            Htmp = Htmp[:, 0, :, :, :] + 1j * Htmp[:, 1, :, :, :]

            Htmp = np.transpose(Htmp, (0, 3, 2, 1))

            if np.size(H) == 0:

                H = Htmp

            else:

                H = np.concatenate((H, Htmp), axis=0)

        H = H.astype(np.complex64) # 默认读取为Complex128精度，转换为Complex64降低存储开销

       

        csi_file = PathRaw + '/' + Prefix + 'InputData' + na + '.npy'

        np.save(csi_file, H) # 首次读取后可存储为npy格式的数据文件

        # H = np.load(csi_file) # 后续可以直接load数据文件

       

        tStart = time.perf_counter()

       

        # 计算并输出定位位置

        print('Calculating localization results')

        result = calcLoc(H, anch_pos, bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num) # 核心定位功能函数，由参赛者实现

       

        # 回填锚点位置信息

        for idx in range(anch_samp_num):

            rowIdx = int(anch_pos[idx][0] - 1)

            result[rowIdx] = np.array([anch_pos[idx][1], anch_pos[idx][2]])

 

        # 输出结果：各位参赛者注意输出值的精度

        print('Writing output position file')

        with open(PathRaw + '/' + Prefix + 'OutputPos' + na + '.txt', 'w') as f:

            np.savetxt(f, result, fmt='%.4f %.4f')

 

        # 统计时间

        tEnd = time.perf_counter()

        print("Total time consuming = {}s\n\n".format(round(tEnd - tStart, 3)))