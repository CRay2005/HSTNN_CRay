import torch
import math
import copy
from torch.autograd import Variable
import numpy as np
from progress.bar import Bar

import copy
import time


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i,p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v,v)
    s = s ** 0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads


def hessian_vector_product(gradsH, params, v, stop_criterion=False):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH, params, grad_outputs=v, only_inputs=True, retain_graph = not stop_criterion)
    return hv


def orthnormal(w, v_list):
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)

def hessian_get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def get_trace_hut(model, data, criterion, n_v, batch_size, bptt, ntokens, loader, cuda = True, channelwise = False, layerwise = False):
    """
    compute the trace of hessian using Hutchinson's method
    """
    # criterion = nn.CrossEntropyLoss()
    # n_v: 采样次数(300)
    # batch_size: 批次大小（25）
    # bptt: 序列长度（35）
    # ntokens: 词表大小（10000）

    print("batch_size: {:5d} | bptt: {:5d} | ntokens: {:5d}".format(batch_size, bptt, ntokens))
    
    assert not (channelwise and layerwise)

    # 步骤1：获取一个批次的数据
    inputs, targets = hessian_get_batch(data, 0, bptt)
    # 例如：
    # inputs: "the cat sat on the mat and the dog ran after it"
    # targets: "cat sat on the mat and the dog ran after it"

    if cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    model.eval()
    # 步骤2：初始化隐藏状态
    hidden = model.init_hidden(batch_size)
    # 创建两个全零张量作为RNN的初始隐藏状态
    # hidden1: [20, 650]  # 第一层RNN
    # hidden2: [20, 650]  # 第二层RNN

    # 步骤3：前向传播
    outputs, hidden = model(inputs, hidden)
    # outputs: 预测下一个词的概率分布
    # 例如：对于"the"，模型预测"cat"的概率

    # 步骤4：计算损失
    loss = criterion(outputs.view(-1, ntokens), targets)
    # 计算预测值与真实值之间的交叉熵损失

    # 步骤5：计算梯度
    loss.backward(create_graph = True)    #create_graph = True表示需要创建计算图，这是为了后续计算Hessian矩阵
    # 获取模型参数和对应的梯度
    params, gradsH = get_params_grad(model)
    # params：模型的所有参数列表，包括：
    #     encoder.weight：词嵌入层权重 [10000, 650]
    #     snn_fc1：第一层SNN权重 [650, 650]
    #     rnn_fc1：第一层RNN权重 [650, 650]
    #     rnn_fv1：第一层RNN循环权重 [650, 650]
    #     snn_fc2：第二层SNN权重 [1300, 650]
    #     rnn_fc2：第二层RNN权重 [1300, 650]
    #     rnn_fv2：第二层RNN循环权重 [650, 650]
    #     decoder.weight：输出层权重 [10000, 1300]
    #     decoder.bias：输出层偏置 [10000]
    # gradsH：对应的梯度列表，形状与参数相同
    
    for p in params:
        print(p.shape, p.size(-1))
    #初始化迹的存储结构
    if channelwise:
        trace_vhv = [[[] for c in range(p.size(-1))] for p in params]
    elif layerwise:
        trace_vhv = [[] for p in params]
    else:
        trace_vhv = []

    # print(np.array(trace_vhv).shape)

    bar = Bar('Computing trace', max=n_v)

    # 步骤6：Hutchinson方法计算Hessian迹
    for i in range(n_v):
        start_time = time.time()
        bar.suffix = f'({i + 1}/{n_v}) |ETA: {bar.elapsed_td}<{bar.eta_td}'
        bar.next()
        # 生成随机向量v
        v = [torch.randint_like(p, high = 2, device = device).float() * 2 - 1 for p in params]
        # v = [
        #     [1.0, -1.0, 1.0, -1.0, ...],
        #     [-1.0, 1.0, -1.0, 1.0, ...],
        #     ...
        # ]
        # 这是Hutchinson方法的一部分
        # 生成的随机向量v满足：每个元素是-1或1；期望值为0；方差为1

        # 计算Hessian向量积
        Hv = hessian_vector_product(gradsH, params, v, stop_criterion= (i==(n_v-1)))
        # 这个计算是Hutchinson方法的核心，它允许我们高效地估计Hessian矩阵的迹，而不需要显式计算完整的Hessian矩阵。这对于分析模型的优化景观和稳定性非常重要
        
        
        # 将Hv和v转换为CPU张量
        # Hv = [Hvi.detach().cpu() for Hvi in Hv]
        # v = [vi.detach().cpu() for vi in v]
        Hv = [Hvi.detach() for Hvi in Hv]
        v = [vi.detach() for vi in v]

        # 计算迹
        with torch.no_grad():
            import copy
            # 按通道计算
            if channelwise:
                for Hv_i in range(len(Hv)):
                    # 计算每个通道的迹
                    for channel_i in range(Hv[Hv_i].size(-1)):
                        dims = len(list(Hv[Hv_i].size()))
                        tmp_hv = copy.deepcopy(Hv[Hv_i])
                        tmp_v  = copy.deepcopy(v[Hv_i])
                        if dims == 2:   #SNN/RNN
                            tmp_hv = tmp_hv.permute(1, 0)
                            tmp_v  = tmp_v.permute(1, 0)
                        elif dims == 3: #LSTM
                            tmp_hv = tmp_hv.permute(2, 0, 1)
                            tmp_v  = tmp_v.permute(2, 0, 1)                            
                        trace_vhv[Hv_i][channel_i].append(tmp_hv[channel_i].flatten().dot(tmp_v[channel_i].flatten()).item())
                        del tmp_hv
                        del tmp_v   
            # 按层计算
            elif layerwise:
                for Hv_i in range(len(Hv)):
                    trace_vhv[Hv_i].append(Hv[Hv_i].flatten().dot(v[Hv_i].flatten()).item())
            # 整体计算
            else:
                trace_vhv.append(group_product(Hv, v).item())
    bar.finish()
    return trace_vhv