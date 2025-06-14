import os
import subprocess

import torch
import time

path = "./"

gpus = 1

available = [i for i in range(gpus)]
seed_list = [1111 * i for i in range(1, 2)]

cnt = 0

def multi_task(path, parameters):
    global cnt
    for r in parameters["ratio"]:
        for se in parameters["seed"]:
            for mo in parameters["model"]:
                my_cmd  = "cd " + path + " && "
                my_cmd += "python rnn-ptb.py "
                my_cmd += "--ratio " + str(r) + " "
                my_cmd += "--seed " + str(se) + " "
                my_cmd += "--model " + str(mo) + " "
                for k in parameters.keys():
                    if k.find("ratio") >= 0 or k.find("seed") >= 0 or k.find("model") >= 0:
                        continue
                    my_cmd += "--" + k + " " + str(parameters[k]) + " "
                
                my_cmd += "--gpu " + str(int(available[cnt % len(available)]))
                cnt += 1
                # subprocess.Popen(my_cmd, shell=True, stdout=None)

                # 将subprocess.Popen改为subprocess.run，实现串行执行
                print(f"Executing: ratio={r}, seed={se}, model={mo}")
                # 清理显存
                print(f"GPU memory before task: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                torch.cuda.empty_cache()

                subprocess.run(my_cmd, shell=True, check=True)
                print(f"GPU memory after task: {torch.cuda.memory_allocated()/1024**2:.2f}MB")

                time.sleep(2)  # 增加到2秒


PTB_XNN_Parameter = {}
PTB_XNN_Parameter["stage1_epoch"] = 200
PTB_XNN_Parameter["stage1_lr"]    = 0.5
PTB_XNN_Parameter["stage1_decay"] = 25
PTB_XNN_Parameter["dropout"]      = 0.5
PTB_XNN_Parameter["nhid"]         = 650
PTB_XNN_Parameter["data"]         = "../../data/penn-treebank"
PTB_XNN_Parameter["mode"]         = "test1"    # "train1" for train and "test1" for test
PTB_XNN_Parameter["emsize"]       = 650
PTB_XNN_Parameter["bptt"]         = 35
PTB_XNN_Parameter["batch_size"]   = 25
PTB_XNN_Parameter["seed"]         = seed_list
PTB_XNN_Parameter["ratio"]        = [0]
PTB_XNN_Parameter["model"]        = ["rnn", "snn", "hybrid"]

# RNN/SNN-Train/Test and HSTNN-Adaptation-Train/Test
#multi_task(path, PTB_XNN_Parameter)

#ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#ratio_list = [0.2040]   # rnn0.20 snn0.4
#ratio_list = [0.2010]   # rnn0.20 snn1.0
ratio_list = [0.9]   


PTB_FFS_Parameter = {}
PTB_FFS_Parameter["stage1_epoch"] = 200
PTB_FFS_Parameter["stage1_lr"]    = 0.5
PTB_FFS_Parameter["stage1_decay"] = 25
PTB_FFS_Parameter["dropout"]      = 0.5
PTB_FFS_Parameter["nhid"]         = 650
PTB_FFS_Parameter["data"]         = "../../data/penn-treebank"
PTB_FFS_Parameter["mode"]         = "test1"    # "train1" for train and "test1" for test
PTB_FFS_Parameter["emsize"]       = 650
PTB_FFS_Parameter["bptt"]         = 35
PTB_FFS_Parameter["batch_size"]   = 25
PTB_FFS_Parameter["seed"]         = seed_list
PTB_FFS_Parameter["ratio"]        = ratio_list
PTB_FFS_Parameter["model"]        = ["ffs"]

# FFS-Train/Test
#multi_task(path, PTB_FFS_Parameter)

PTB_HBR_Parameter = {}
PTB_HBR_Parameter["stage2_epoch"] = 200
PTB_HBR_Parameter["stage2_lr"]    = 0.1
PTB_HBR_Parameter["stage2_decay"] = 25
PTB_HBR_Parameter["dropout"]      = 0.5
PTB_HBR_Parameter["nhid"]         = 650
PTB_HBR_Parameter["data"]         = "../../data/penn-treebank"
PTB_HBR_Parameter["mode"]         = "train2"
PTB_HBR_Parameter["emsize"]       = 650
PTB_HBR_Parameter["bptt"]         = 35
PTB_HBR_Parameter["batch_size"]   = 25
PTB_HBR_Parameter["seed"]         = seed_list
PTB_HBR_Parameter["ratio"]        = ratio_list
PTB_HBR_Parameter["model"]        = ["hybrid"]

# HSTNN-Restoration-Train/Test
multi_task(path, PTB_HBR_Parameter)

