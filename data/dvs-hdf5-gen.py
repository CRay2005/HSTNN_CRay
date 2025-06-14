import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dcll'))
from dcll.pytorch_libdcll import *
from dcll.experiment_tools import *
from dcll.load_dvsgestures_sparse import *
import torch
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(1111)

# 使用绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(current_dir, 'DvsGesture', 'dvs_gestures_events.hdf5')

print("Current working directory:", os.getcwd())
print("Target filename:", filename)
print("File exists:", os.path.exists(filename))

create_events_hdf5(filename)

