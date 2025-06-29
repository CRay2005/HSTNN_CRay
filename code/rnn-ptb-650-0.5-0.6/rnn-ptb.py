# coding: utf-8
import argparse
import time
import math
import os
import copy
import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

import random

import data

torch.multiprocessing.set_start_method('spawn')
torch.set_num_threads(16)

###############################################################################
# Define Activation Function for SNNs
###############################################################################
lens = 0.5
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float() / (2 * lens)

WinFunc = ActFun.apply

###############################################################################
# Define HNN Model
###############################################################################

class HNNModel(nn.Module):

    def __init__(self, ntoken, ninp, rnn_shape, snn_shape, dropout=None, device="cuda", func="window", union=False):
        super(HNNModel, self).__init__()
        
        assert len(rnn_shape) == 2 and len(snn_shape) == 2
        assert rnn_shape[0] + snn_shape[0] > 0 and rnn_shape[1] + snn_shape[1] > 0

        self.ntoken    = ntoken
        self.ninp      = ninp
        self.rnn_shape = rnn_shape
        self.snn_shape = snn_shape
        self.func      = func
        self.dropout   = nn.Dropout(dropout)
        self.device    = torch.device(device)
        self.union     = union
        
        # constant parameters
        self.nlayers = 2
        self.decay   = 0.6
        self.thresh  = 0.6
        
        if self.func == "window":
            self.act_fun = WinFunc

        # weights tensors
        self.encoder = nn.Embedding(ntoken, ninp)
        
        if self.union is False:
            if self.snn_shape[0] > 0:
                self.snn_fc1 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(ninp, snn_shape[0])))
            if self.snn_shape[1] > 0:
                self.snn_fc2 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(snn_shape[0] + rnn_shape[0], snn_shape[1])))
            if self.rnn_shape[0] > 0:
                self.rnn_fc1 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(ninp, rnn_shape[0])))
            if self.rnn_shape[1] > 0:
                self.rnn_fc2 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(snn_shape[0] + rnn_shape[0], rnn_shape[1]))) # caution!
            if self.rnn_shape[0] > 0:    
                self.rnn_fv1 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(rnn_shape[0], rnn_shape[0])))
            if self.rnn_shape[1] > 0:
                self.rnn_fv2 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(rnn_shape[1], rnn_shape[1])))
        else:
            if self.snn_shape[0] > 0:
                self.snn1 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(ninp, snn_shape[0])))
            if self.snn_shape[1] > 0:
                self.snn2 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(snn_shape[0] + rnn_shape[0], snn_shape[1])))
            if self.rnn_shape[0] > 0:
                self.rnn1 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(ninp + rnn_shape[0], rnn_shape[0])))
            if self.rnn_shape[1] > 0:
                self.rnn2 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(snn_shape[0] + rnn_shape[0] + rnn_shape[1], rnn_shape[1]))) # caution!
        self.decoder = nn.Linear(snn_shape[1] + rnn_shape[1], ntoken) # caution!
        
        self.init_weights()

    def init_weights(self):
        # initialize the weights
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        # initialize the hidden tensors to zero
        hidden1 = torch.zeros([bsz, self.rnn_shape[0]], dtype=torch.float32, device=self.device) 
        hidden2 = torch.zeros([bsz, self.rnn_shape[1]], dtype=torch.float32, device=self.device)      
        return (hidden1, hidden2)    

    def load_union_state(self, arg_dict):
        rnn_type = "rnn"
        for k in arg_dict.keys():
            if k.find("lstm") >= 0:
                rnn_type = "lstm"
        if rnn_type == "rnn":    
            rnn1 = torch.cat((arg_dict["rnn_fc1"], arg_dict["rnn_fv1"]), dim=0)
            rnn2 = torch.cat((arg_dict["rnn_fc2"], arg_dict["rnn_fv2"]), dim=0)
        else:
            rnn1 = torch.cat((arg_dict["lstm_wi1"], arg_dict["lstm_wh1"]), dim=1)
            rnn2 = torch.cat((arg_dict["lstm_wi2"], arg_dict["lstm_wh2"]), dim=1)
        arg_dict["snn1"] = copy.deepcopy(arg_dict["snn_fc1"])
        arg_dict["snn2"] = copy.deepcopy(arg_dict["snn_fc2"])                        
        union_dict = {k: v for k, v in arg_dict.items() if k in self.state_dict().keys()}
        union_dict[rnn_type + "1"] = rnn1
        union_dict[rnn_type + "2"] = rnn2
        self.load_state_dict(union_dict)

    def rnn_update(self, fc, fv, inputs, last_state):
        state = inputs.mm(fc) + last_state.mm(fv)
        activation = state.sigmoid()
        return activation
    
    def rnn_union_update(self, uf, inputs, last_state):
        ui    = torch.cat((inputs, last_state), dim=1)
        state = ui.mm(uf)
        activation = state.sigmoid()
        return activation


    def snn_update(self, fc, inputs, mem, spike):
        state = inputs.mm(fc)
        mem = mem * (1 - spike) * self.decay + state
        now_spike = self.act_fun(mem - self.thresh)
        return mem, now_spike.float()    


    def forward(self, raw_input, hidden):
        
        # input vector embedding
        emb = self.encoder(raw_input)
        input = self.dropout(emb)
        
        # embedded inputs forward pass
        n_win, batch_size, input_size = input.size()
        
        h1_mem = h1_spike = torch.zeros(batch_size, self.snn_shape[0], device = self.device)
        h2_mem = h2_spike = torch.zeros(batch_size, self.snn_shape[1], device = self.device)
        h1_y, h2_y = hidden
        
        buf = []
        if self.union is False:
            for t in range(n_win):
                output0 = input[t]
                if self.snn_shape[0] > 0:
                    h1_mem, h1_spike = self.snn_update(self.snn_fc1, output0, h1_mem, h1_spike)
                if self.rnn_shape[0] > 0:
                    h1_y             = self.rnn_update(self.rnn_fc1, self.rnn_fv1, output0, h1_y)
                output1          = torch.cat((h1_spike, h1_y), dim=1)
                
                if self.snn_shape[1] > 0:
                    h2_mem, h2_spike = self.snn_update(self.snn_fc2, output1, h2_mem, h2_spike)
                if self.rnn_shape[1] > 0:
                    h2_y             = self.rnn_update(self.rnn_fc2, self.rnn_fv2, output1, h2_y)
                output2          = torch.cat((h2_spike, h2_y), dim=1)
            
                buf.append(output2)
        else:
            for t in range(n_win):
                output0 = input[t]      #关键，时间步循环
                if self.snn_shape[0] > 0:
                    h1_mem, h1_spike = self.snn_update(self.snn1, output0, h1_mem, h1_spike)
                if self.rnn_shape[0] > 0:
                    h1_y             = self.rnn_union_update(self.rnn1, output0, h1_y)
                output1          = torch.cat((h1_spike, h1_y), dim=1)
                
                if self.snn_shape[1] > 0:
                    h2_mem, h2_spike = self.snn_update(self.snn2, output1, h2_mem, h2_spike)
                if self.rnn_shape[1] > 0:
                    h2_y             = self.rnn_union_update(self.rnn2, output1, h2_y)
                output2          = torch.cat((h2_spike, h2_y), dim=1)
            
                buf.append(output2)      #收集所有时间步的输出      

        stacked_output = torch.stack(buf, dim=0)

        # dropout and decoded
        output  = self.dropout(stacked_output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))

        return decoded.view(output.size(0), output.size(1), decoded.size(1)), (h1_y, h2_y)       


###############################################################################
# Parse arguments
###############################################################################

parser = argparse.ArgumentParser(description='HNN Model on Language Dataset')

parser.add_argument('--data', type=str,
                    default='../../data/penn-treebank',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='hybrid', help='type of network (rnn, snn, ffs[directly hybrid], hybrid)')

parser.add_argument('--mode', type=str, default='train1', help='type of operation (train1, test1, train2, test2)')
# Activate parameters
parser.add_argument('--batch_size', type=int, default=25, metavar='N', help='batch size')
parser.add_argument('--stage1_epochs', type=int, default=150, help='training epochs for Adaptation stage')
parser.add_argument('--stage2_epochs', type=int, default=150, help='training epochs for Restoration stage')

parser.add_argument('--stage1_lr', type=float, default=0.5, help='learning rate for Adaptation stage')
parser.add_argument('--stage2_lr', type=float, default=0.075, help='learning rate for Restoration stage')

parser.add_argument('--stage1_decay', type=int, default=25, help='lr decay for Adaptation stage')
parser.add_argument('--stage2_decay', type=int, default=25, help='lr decay for Restoration stage')

parser.add_argument('--ratio', type=float, default=0.5, help='snn ratio of the hybird network')

# Default parameters
parser.add_argument('--emsize', type=int, default=650, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=120, help='number of hidden units per layer')
parser.add_argument('--bptt', type=int, default=25, help='sequence length')
parser.add_argument('--clip', type=float, default=1, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='report interval')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--gpu', type=str, default='0', help='gpu number')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.manual_seed(args.seed)
device = torch.device("cuda")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(args.seed)


###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5

    print('reset optimizer with the learning rate', optimizer.param_groups[0]['lr'])
    return optimizer

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target




eval_batch_size = args.batch_size
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

ntokens = len(corpus.dictionary)

criterion = nn.CrossEntropyLoss()
l1_criterion = nn.L1Loss()


def train(arg_model, optimizers, epoch, logger):
    arg_model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    #1.初始化隐藏状态
    hidden = arg_model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        #2.截断梯度历史
        hidden = repackage_hidden(hidden)
        #3.梯度清零
        arg_model.zero_grad()
        #4.前向传播
        output, hidden = arg_model.forward(data, hidden)
        #5.计算损失
        target_loss = criterion(output.view(-1, ntokens), targets)
        both_loss = target_loss
        #6.BPTT反向传播（替代梯度自动生效）
        #虽然代码中只写了 backward()，但PyTorch根据模型的时序特性自动执行了BPTT算法！
        both_loss.backward()
        #7.梯度裁剪
        torch.nn.utils.clip_grad_norm_(arg_model.parameters(), args.clip)
        #8.更新参数
        optimizers.step()
        optimizers.zero_grad()

        total_loss += target_loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed  = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | {:5.2f} ms/batch | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(arg_model, data_source):
    # Turn on evaluation mode which disables dropout.
    arg_model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = arg_model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = arg_model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

def train_over_epoch(arg_model, epoch_num, optimizer, lr, decay_epoch, logger, arg_model_name):
    # record variables
    valid_loss_record = []
    best_val_loss = None
    best_model = None
    # train epoch_num epochs
    for epoch in range(1, epoch_num + 1):
        epoch_start_time = time.time()
        train(arg_model, optimizer, epoch, logger)
        val_loss = evaluate(arg_model, val_data)
        valid_loss_record.append(copy.deepcopy(math.exp(val_loss)))
        logger.info('-' * 89)
        logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        logger.info('-' * 89)

        optimizer = lr_scheduler(optimizer, epoch, init_lr=lr, lr_decay_epoch=decay_epoch)

        if not best_val_loss or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(arg_model)
            state = {
                'net' : arg_model.state_dict(),
                'seed': args.seed,
                'rnn_shape': arg_model.rnn_shape,
                'snn_shape': arg_model.snn_shape,
                'ntoken'   : arg_model.ntoken,
                'ninp'     : arg_model.ninp,
                'snn_func' : arg_model.func,
                'val_loss_record': valid_loss_record
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/' + arg_model_name + ".t7")
    test_trained_model(best_model, logger)
    return best_model

def test_trained_model(arg_model, logger):
    test_loss = evaluate(arg_model, test_data)
    logger.info('=' * 89)
    logger.info('| Performance on Test Set | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    logger.info('=' * 89)
      

def train_origin_model(model_name, logger):
    if args.model == "rnn":
        rnn_shape = [args.nhid, args.nhid]
        snn_shape = [0, 0]
    elif args.model == "snn":
        rnn_shape = [0, 0]
        snn_shape = [args.nhid, args.nhid]
    elif args.model == "hybrid":
        rnn_shape = [args.nhid, args.nhid]
        snn_shape = [args.nhid, args.nhid]
    else:   # learning from scratch
        rnn_shape = [int((1 - args.ratio) * args.nhid), int((1 - args.ratio) * args.nhid)]
        snn_shape = [int(args.ratio * args.nhid),       int(args.ratio * args.nhid)]
    
    logger.info("model with rnn_shape: [{:3d}, {:3d}]".format(rnn_shape[0], rnn_shape[1]))
    logger.info("model with snn_shape: [{:3d}, {:3d}]".format(snn_shape[0], snn_shape[1]))

    origin_model = HNNModel(ntokens, args.emsize, rnn_shape, snn_shape, args.dropout, "cuda").to(device)
    optimizer    = optim.SGD(origin_model.parameters(), lr=args.stage1_lr, momentum=0.9)

    trained_model = train_over_epoch(origin_model, args.stage1_epochs, optimizer, args.stage1_lr, args.stage1_decay, logger, model_name)
    return trained_model

def train_pruned_model(arg_model, model_name, logger):
    optimizer = optim.SGD(arg_model.parameters(), lr=args.stage2_lr, momentum=0.9)
    trained_model = train_over_epoch(arg_model, args.stage2_epochs, optimizer, args.stage2_lr, args.stage2_decay, logger, model_name)
    return trained_model

def print_eff_index(name, l, logger):
    st = "["
    for i in l:
        st += str(i) + ", "
    st = st[:-2] + "]"
    logger.info(name + ": " + st)


def compute_effective_indices(arg_model, logger, hessian_mode="trace"):
    assert args.model == "hybrid"
    from hessian_pruner import HessianPruner
    if not os.path.isdir('traces'):
                os.system('mkdir -p traces')
    # todo: func!!!
    trace_file_name = "./traces/" + "trace" + "_" + "hybrid" + "_"  + str(args.seed) + ".npy"
    pruner = HessianPruner(arg_model, trace_file_name, hessian_mode=hessian_mode)
    eff_dict = pruner.make_pruned_model(train_data, criterion, device, args.ratio, args.seed, args.batch_size, args.bptt, ntokens)

    print_eff_index("rnn1", sorted(eff_dict["rnn1"]), logger)
    print_eff_index("rnn2", sorted(eff_dict["rnn2"]), logger)
    print_eff_index("snn1", sorted(eff_dict["snn1"]), logger)
    print_eff_index("snn2", sorted(eff_dict["snn2"]), logger)

    logger.info("rnn_layer[0]: " + str(len(eff_dict["rnn1"])))
    logger.info("rnn_layer[1]: " + str(len(eff_dict["rnn2"])))
    logger.info("snn_layer[0]: " + str(len(eff_dict["snn1"])))
    logger.info("snn_layer[1]: " + str(len(eff_dict["snn2"])))
    
    return eff_dict
    


def logger_generation(file_name):
    if not os.path.isdir('log'):
        os.mkdir('log')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)

    fh = logging.FileHandler("./log/" + file_name + ".log")
    fh.setLevel(logging.DEBUG)
    
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

def load_model(arg_model_name, make_union=False):
    ckpt_path = "./checkpoint/" + arg_model_name + ".t7"
    ckpts     = torch.load(ckpt_path, map_location="cpu")
    weights_dict  = ckpts["net"]
    tmp_rnn_shape = ckpts["rnn_shape"]
    tmp_snn_shape = ckpts["snn_shape"]
    tmp_ntoken    = ckpts["ntoken"]
    tmp_ninp      = ckpts["ninp"]
    # todo: func!!!
    tmp_func      = "window"
    tmp_model     = HNNModel(ntoken=tmp_ntoken, ninp=tmp_ninp, rnn_shape=tmp_rnn_shape, snn_shape=tmp_snn_shape, dropout=args.dropout, device=device, func=tmp_func, union=make_union).to(device)
    if make_union == False:
        tmp_model.load_state_dict(weights_dict)
    else:
        tmp_model.load_union_state(weights_dict)
    return tmp_model 

def weight_shrink(arg_weight, eff_row = None, row_size = None, eff_col = None, col_size = None):
    assert arg_weight is not None
    tmp_weight = copy.deepcopy(arg_weight).to("cpu")
    if row_size is not None:
        assert tmp_weight.shape[0] == row_size
        left_matrix = torch.zeros(len(eff_row), row_size)
        for per_row in range(len(eff_row)):
            left_matrix[per_row][eff_row[per_row]] = 1
        cc = torch.mm(left_matrix, tmp_weight)
        tmp_weight = copy.deepcopy(torch.mm(left_matrix, tmp_weight))
    if col_size is not None:
        assert tmp_weight.shape[1] == col_size
        right_matrix = torch.zeros(col_size, len(eff_col))
        for per_col in range(len(eff_col)):
            right_matrix[eff_col[per_col]][per_col] = 1
        tmp_weight = copy.deepcopy(torch.mm(tmp_weight, right_matrix))
    return tmp_weight




def shrink(arg_model, index_dict):
    pruned_model = copy.deepcopy(arg_model).to("cpu")
    old_dict = arg_model.state_dict()
    # print(arg_model.state_dict().keys())
    en_w     = old_dict["encoder.weight"] # (raw_input_length, embedded_size)
    de_w     = old_dict["decoder.weight"] # (raw_output_length, rnn_layer[1] + snn_layer[1])
    de_b     = old_dict["decoder.bias"]   # (raw_output_length)
    
    snn_fc1     = old_dict["snn_fc1"]     # (embbeded_size, snn_layer[0])
    rnn_fc1     = old_dict["rnn_fc1"]     # (embbeded_size, rnn_layer[0])
    rnn_fv1     = old_dict["rnn_fv1"]     # (rnn_layer[0], rnn_layer[0])

    snn_fc2     = old_dict["snn_fc2"]     # (snn_layer[0] + rnn_layer[0], snn_layer[1]) 
    rnn_fc2     = old_dict["rnn_fc2"]     # (snn_layer[0] + rnn_layer[0], rnn_layer[1])
    rnn_fv2     = old_dict["rnn_fv2"]     # (rnn_layer[1], rnn_layer[1])

    snn_fc1_eff_col = sorted(index_dict["snn1"])
    rnn_fc1_eff_col = sorted(index_dict["rnn1"])
    rnn_fv1_eff_col = sorted(index_dict["rnn1"])
    rnn_fv1_eff_row = sorted(index_dict["rnn1"])

    snn_fc2_eff_col = sorted(index_dict["snn2"])
    rnn_fc2_eff_col = sorted(index_dict["rnn2"])
    rnn_fv2_eff_col = sorted(index_dict["rnn2"])
    rnn_fv2_eff_row = sorted(index_dict["rnn2"])

    # mention!
    incre1 = arg_model.snn_shape[0]
    tmp1 = [index + incre1 for index in index_dict["rnn1"]]
    snn_fc2_eff_row = sorted(index_dict["snn1"] + tmp1)
    rnn_fc2_eff_row = snn_fc2_eff_row

    incre2 = arg_model.snn_shape[1]
    tmp2 = [index + incre2 for index in index_dict["rnn2"]]
    de_w_eff_col = sorted(index_dict["snn2"] + tmp2)

    snn_fc1_new = weight_shrink(snn_fc1, eff_row=None, row_size=None,
                                         eff_col=snn_fc1_eff_col, col_size=arg_model.snn_shape[0])
    rnn_fc1_new = weight_shrink(rnn_fc1, eff_row=None, row_size=None,
                                         eff_col=rnn_fc1_eff_col, col_size=arg_model.rnn_shape[0])
    rnn_fv1_new = weight_shrink(rnn_fv1, eff_row=rnn_fv1_eff_row, row_size=arg_model.rnn_shape[0],
                                         eff_col=rnn_fv1_eff_col, col_size=arg_model.rnn_shape[0])
    
    snn_fc2_new = weight_shrink(snn_fc2, eff_row=snn_fc2_eff_row, row_size=arg_model.rnn_shape[0] + arg_model.snn_shape[0],
                                         eff_col=snn_fc2_eff_col, col_size=arg_model.snn_shape[1])
    rnn_fc2_new = weight_shrink(rnn_fc2, eff_row=rnn_fc2_eff_row, row_size=arg_model.rnn_shape[0] + arg_model.snn_shape[0],
                                         eff_col=rnn_fc2_eff_col, col_size=arg_model.rnn_shape[1])
    rnn_fv2_new = weight_shrink(rnn_fv2, eff_row=rnn_fv2_eff_row, row_size=arg_model.rnn_shape[1],
                                         eff_col=rnn_fv2_eff_col, col_size=arg_model.rnn_shape[1])    

    de_w_new    = weight_shrink(de_w, eff_row=None, row_size=None, eff_col=de_w_eff_col, 
                                         col_size=arg_model.rnn_shape[1] + arg_model.snn_shape[1])
    

    pruned_model.rnn_shape = [len(index_dict["rnn1"]), len(index_dict["rnn2"])]
    pruned_model.snn_shape = [len(index_dict["snn1"]), len(index_dict["snn2"])]
    pruned_model.val_loss_record = []

    pruned_model.encoder.weight.data = copy.deepcopy(en_w).to("cuda")
    pruned_model.snn_fc1.data = copy.deepcopy(snn_fc1_new).to("cuda")
    pruned_model.snn_fc2.data = copy.deepcopy(snn_fc2_new).to("cuda")
    pruned_model.rnn_fc1.data = copy.deepcopy(rnn_fc1_new).to("cuda")
    pruned_model.rnn_fc2.data = copy.deepcopy(rnn_fc2_new).to("cuda")
    pruned_model.rnn_fv1.data = copy.deepcopy(rnn_fv1_new).to("cuda")
    pruned_model.rnn_fv2.data = copy.deepcopy(rnn_fv2_new).to("cuda")
    pruned_model.decoder.weight.data = copy.deepcopy(de_w_new).to("cuda")
    pruned_model.decoder.bias.data = copy.deepcopy(de_b).to("cuda")
    pruned_model.to("cuda")
    return pruned_model


def get_pruned_model(model_name, logger, union=False, hessian_mode="trace"):
    union_model = load_model(model_name, make_union=union)
    test_trained_model(union_model, logger)
    eff_dict    = compute_effective_indices(union_model, logger, hessian_mode=hessian_mode)
    del union_model
    trained_model = load_model(model_name, make_union=False)
    pruned_model = shrink(trained_model, eff_dict)
    test_trained_model(pruned_model, logger)
    del trained_model
    torch.cuda.empty_cache()
    return pruned_model

assert args.model in ["rnn", "snn", "ffs", "hybrid"]

if args.data.find("wiki") >= 0:
    dataset_name = "wiki"
else:
    assert args.data.find("penn") >= 0 or args.data.find("ptb") >= 0
    dataset_name = "ptb"


if args.mode == "train1":
    model_name = dataset_name + "_" + args.model + "_" + str(args.seed)
    if args.model == "ffs":
        model_name += "_" + str(args.ratio)
    logfile_name = "train" + "_" + model_name

    train1_logger = logger_generation(logfile_name)
    trained_model = train_origin_model(model_name, train1_logger)

elif args.mode == "test1":
    model_name = dataset_name + "_" + args.model + "_" + str(args.seed)
    if args.model == "ffs":
        model_name += "_" + str(args.ratio)
    logfile_name = "test" + "_" + model_name

    test1_logger  = logger_generation(logfile_name)
    trained_model = load_model(model_name, make_union=False)
    print(trained_model.rnn_shape, trained_model.snn_shape)
    test_trained_model(trained_model, test1_logger)

elif args.mode == "train2":
    assert args.model == "hybrid"
    if args.model == "hybrid":
        input_model_name  = dataset_name + "_" + args.model + "_" + str(args.seed)
        pruned_model_name = dataset_name + "_" + args.model + "_" + str(args.seed) + "_" + str(args.ratio)
        logfile_name = "train" + "_" + pruned_model_name
        train2_logger = logger_generation(logfile_name)
        pruned_model = get_pruned_model(input_model_name, train2_logger, union=True) # caution!
        trained_model = train_pruned_model(pruned_model, pruned_model_name, train2_logger)
    else:
        print("Unknown Mode")
        assert False

elif args.mode == "test2":
    assert args.model == "hybrid"
    if args.model == "hybrid":
        model_name = dataset_name + "_" + args.model + "_" + str(args.seed) + "_" + str(args.ratio)
        logfile_name = "test" + "_" + model_name
        test2_logger = logger_generation(logfile_name)
        final_model  = load_model(model_name, make_union=False)
        print(final_model.rnn_shape, final_model.snn_shape)
        test_trained_model(final_model, test2_logger)
    else:
        print("Unknown Mode")
        assert False
else:
    print("Unknown Mode")
    assert False
