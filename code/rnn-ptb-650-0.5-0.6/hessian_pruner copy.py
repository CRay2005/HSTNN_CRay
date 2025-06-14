import torch
import torch.nn as nn
from collections import OrderedDict
from utils.kfac_utils import fetch_mat_weights
# from utils.common_utils import (tensor_to_list, PresetLRScheduler)
from utils.prune_utils import (filter_indices,
                               filter_indices_ni,
                               get_threshold,
                               update_indices,
                               normalize_factors,
                               prune_model_ni)
# from utils.network_utils import stablize_bn
from tqdm import tqdm

from hessian_fact import get_trace_hut
from pyhessian.hessian import hessian
from pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal

import numpy as np
import time
import scipy.linalg
import os.path
from os import path


class HessianPruner:
        def __init__(self,
                     model,
                     trace_file_name,
                     fix_layers=0,
                     hessian_mode='trace'):
            self.iter = 0
            #self.known_modules = {'Linear', 'Conv2d'}
            self.modules = []
            self.model = model
            self.fix_layers = fix_layers

            self.W_pruned = {}
            self.S_l = None

            self.hessian_mode = hessian_mode
            self.trace_file_name = trace_file_name

            self.importances = {}
            self._inversed = False
            self._cfgs = {}
            self._indices = {}


        # top function
        def make_pruned_model(self, dataloader, criterion, device, snn_ratio, seed, batch_size, bptt, ntokens, is_loader=False, normalize=True, re_init=False, n_v=300):
            self.snn_ratio = snn_ratio # use for some special case, particularly slq_full, slq_layer
            self.seed = seed
            self._prepare_model()
            self.mask_list = self._compute_hessian_importance(dataloader, criterion, device, batch_size, bptt, ntokens,  is_loader, n_v=n_v)
            print("Finished Hessian Importance Computation!")
            return self.mask_list

        def _prepare_model(self):
            count = 0
            for it in self.model.named_parameters():
                if it[0].find("all_fc") >= 0:
                    continue
                if it[0].find("wh") >= 0:
                    continue
                if it[0].find("decoder") >= 0:
                    continue
                if it[0].find("fv") >= 0:
                    continue
                if it[0].find("encoder") >= 0:
                    continue
                self.modules.append(it)


        def _compute_hessian_importance(self, dataloader, criterion, device, batch_size, bptt, ntokens, is_loader, n_v=300):
            print("is_loader", is_loader)
            ###############
            # Here, we use the fact that Conv does not have bias term
            ###############
            if self.hessian_mode == 'trace':
                # 1.只对特定层（SNN和RNN）计算Hessian
                for k, v in self.model.named_parameters():
                    if k.find("all_fc") >= 0:
                        v.requires_grad = False
                    elif k.find("wh") >= 0:
                        v.requires_grad = False
                    elif k.find("decoder") >= 0:
                        v.requires_grad = False
                    elif k.find("fv") >= 0:
                        v.requires_grad = False
                    elif k.find("encoder") >= 0:
                        v.requires_grad = False
                    else:
                        print(k, v.requires_grad)
                # 2.加载或者调用get_trace_hut函数计算Hessian
                trace_dir = self.trace_file_name
                print(trace_dir)
                if os.path.exists(trace_dir):
                    print(f"Loading trace from {trace_dir}")
                    import numpy as np
                    results = np.load(trace_dir, allow_pickle=True)
                else:
                    results = get_trace_hut(self.model, dataloader, criterion, n_v, batch_size, bptt, ntokens, loader=is_loader, channelwise=True, layerwise=False)
                    import numpy as np
                    np.save(self.trace_file_name, results)

                for m in self.model.parameters():
                    m.requires_grad = True

                #3.通道重要性平均值计算
                channel_trace, weighted_trace = [], []
                # results结构：[层][通道][采样次数]
                #处理层 layer
                for k, layer in enumerate(results):
                    # print(k, layer)
                    channel_trace.append(torch.zeros(len(layer)))
                    weighted_trace.append(torch.zeros(len(layer)))
                    #处理通道 channel
                    for cnt, channel in enumerate(layer):
                        #print(cnt, channel.shape, len(layer))
                        # 计算每个通道的平均值
                        channel_trace[k][cnt] = sum(channel) / len(channel)
                #for i in channel_trace:
                    # print(len(i))
                # print(len(results), self.model.parameters())

                # 4.weight加权重要性计算
                # 重要性 = Hessian迹 × (权重范数²/权重元素数量)
                # Hessian迹反映参数对损失函数的敏感性
                # 权重范数反映参数的大小
                # 结合两者得到更准确的神经元重要性评估
                for k, mod in enumerate(self.modules):
                    tmp = []
                    # k：   层索引（0, 1, 2, 3对应snn_fc1, rnn_fc1, snn_fc2, rnn_fc2）
                    # mod： 每个模块的信息，格式为(模块名称, 模块对象)
                    m = mod[0]
                    import copy 
                    cur_weight = copy.deepcopy(mod[1].data) #mod[1]：模块对象（如Linear层），mod[1].data：该层的权重张量
                    dims = len(list(cur_weight.size()))

                    # 维度转换
                    # 目的：统一权重张量的格式，使第一维度对应通道数（与Hessian迹的通道对应）                
                    if dims == 2:
                        # 2维情况（全连接层）：
                        # 原始形状：[输出神经元数, 输入神经元数]
                        # 转换后：[输入神经元数, 输出神经元数]
                        cur_weight = cur_weight.permute(1, 0)
                    elif dims == 3:
                        # 3维情况（可能的卷积层或特殊结构）：
                        # 原始形状：[输出通道, 输入通道, 其他维度]
                        # 转换后：[其他维度, 输出通道, 输入通道]   
                        cur_weight = cur_weight.permute(2, 0, 1)
                    for cnt, channel in enumerate(cur_weight):
                        # channel_trace[k][cnt]：上面计算得到的，第k层第cnt个通道的平均Hessian迹
                        # .detach()：从计算图中分离，避免梯度计算
                        # .norm()**2：计算L2范数的平方（权重向量的平方和）
                        # / channel.numel()：除以权重元素数量，归一化
                        tmp.append( (channel_trace[k][cnt] * channel.detach().norm()**2 / channel.numel()).cpu().item())
                    print(m, len(tmp))
                    self.importances[str(m)] = (tmp, len(tmp))
                    #self.W_pruned[m] = fetch_mat_weights(m, False)
            else:
                print("Unsupported mode")
                assert False

            tmp_imp_list = list(self.importances.items())

            # 5.混合网络结构处理
            rnn_list = [None, None]     # rnn_fc1, rnn_fc2
            snn_list = [None, None]     # snn_fc1, snn_fc2

            for unit in tmp_imp_list:
                if unit[0].find("rnn") >= 0 or unit[0].find("lstm") >= 0:
                    if unit[0].find("1") >= 0:
                        rnn_list[0] = unit[1][0]        # rnn_fc1层的重要性
                    else:
                        assert unit[0].find("2") >= 0
                        rnn_list[1] = unit[1][0]        # rnn_fc2层的重要性
                elif unit[0].find("snn") >= 0:
                    if unit[0].find("1") >= 0:
                        snn_list[0] = unit[1][0]        # snn_fc1层的重要性
                    else:
                        assert unit[0].find("2") >= 0
                        snn_list[1] = unit[1][0]        # snn_fc2层的重要性
                else:
                    continue

            rnn_shape = [len(rnn_list[0]), len(rnn_list[1])]
            snn_shape = [len(snn_list[0]), len(snn_list[1])]

            # 6.重要性排序
            rnn_tuple_list = []
            snn_tuple_list = []
            # 创建(索引, 重要性)元组列表
            for no in range(len(rnn_list[0])):
                rnn_tuple_list.append((no, rnn_list[0][no]))
            for no in range(len(rnn_list[1])):
                rnn_tuple_list.append((no + rnn_shape[0], rnn_list[1][no]))
            # 按重要性降序排序
            for no in range(len(snn_list[0])):
                snn_tuple_list.append((no, snn_list[0][no]))
            for no in range(len(snn_list[1])):
                snn_tuple_list.append((no + snn_shape[0], snn_list[1][no]))

            sorted_rnn_list = sorted(rnn_tuple_list, key=lambda x:x[1])     #, reverse=True) #按重要性降序排序
            sorted_snn_list = sorted(snn_tuple_list, key=lambda x:x[1])     #, reverse=True) #按重要性降序排序

            sorted_rnn_list.reverse()   #降序排序，如[(3, 0.9)，(0, 0.8)，(4, 0.6)...]
            sorted_snn_list.reverse()   #降序排序
            
            del rnn_list, snn_list, rnn_tuple_list, snn_tuple_list

            # 7.剪枝神经元的数量
            eff_rnns_number = int((rnn_shape[0] + rnn_shape[1]) * (1.0 - self.snn_ratio))
            eff_snns_number = int((snn_shape[0] + snn_shape[1]) * (self.snn_ratio))

            print("!!!!!!!!!!!eff_rnns_number, eff_snns_number minus 10!!!!!!!!!!!!!!!!!!!!")
            # if eff_rnns_number > 30:
            #     eff_rnns_number = eff_rnns_number -10
            # if eff_snns_number > 30:
            #     eff_snns_number = eff_snns_number -10


            rnn_layer_util = [False, False] #使用布尔数组记录每一层是否至少保留了一个神经元
            snn_layer_util = [False, False]

            # 8. RNN、SNN至少保留一个神经元
            # check whether at least one neuron(rnn or snn) exists in every layer
            for idx in range(0, eff_rnns_number):
                if sorted_rnn_list[idx][0] >= rnn_shape[0]:
                    rnn_layer_util[1] = True
                else:
                    rnn_layer_util[0] = True
            
            for idx in range(0, eff_snns_number):
                if sorted_snn_list[idx][0] >= snn_shape[0]:
                    snn_layer_util[1] = True
                else:
                    snn_layer_util[0] = True
            
            # fix the structure
            def not_in_one_layer(idx1, idx2, thres):
                return (idx1 < thres and idx2 >= thres) or (idx2 < thres and idx1 >= thres)
            
            eff_rnns_list = []
            for idx in range(0, eff_rnns_number):
                eff_rnns_list.append(sorted_rnn_list[idx][0])
            
            if rnn_layer_util[0] is False or rnn_layer_util[1] is False:
                last_one = eff_rnns_list[-1]
                for idx in range(eff_rnns_number, rnn_shape[0] + rnn_shape[1]):
                    curr_one = sorted_rnn_list[idx][0]
                    if not_in_one_layer(last_one, curr_one, rnn_shape[0]) is True:
                        eff_rnns_list[-1] = curr_one
                        break

            eff_snns_list = []
            for idx in range(0, eff_snns_number):
                eff_snns_list.append(sorted_snn_list[idx][0])              

            if snn_layer_util[0] is False or snn_layer_util[1] is False:
                last_one = eff_snns_list[-1]
                for idx in range(eff_snns_number, snn_shape[0] + snn_shape[1]):
                    curr_one = sorted_snn_list[idx][0]
                    if not_in_one_layer(last_one, curr_one, snn_shape[0]) is True:
                        eff_snns_list[-1] = curr_one
                        break

            del rnn_layer_util, snn_layer_util

            # 9. 最终输出
            # output
            eff_dict = {}
            eff_dict["rnn1"] = []
            eff_dict["rnn2"] = []
            eff_dict["snn1"] = []
            eff_dict["snn2"] = []

            for item in eff_rnns_list:
                if item < rnn_shape[0]:
                    eff_dict["rnn1"].append(item)
                else:
                    eff_dict["rnn2"].append(item - rnn_shape[0])
            
            for item in eff_snns_list:
                if item < snn_shape[0]:
                    eff_dict["snn1"].append(item)
                else:
                    eff_dict["snn2"].append(item - snn_shape[0])
            # print(sorted_rnn_list)
            # print(eff_dict)
            # print(sorted_snn_list)
            return eff_dict

