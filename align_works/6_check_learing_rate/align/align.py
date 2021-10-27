import numpy as np
import torch
import paddle
from reprod_log import ReprodDiffHelper

diff = ReprodDiffHelper()
info_torch = diff.load_info('../lr_torch.npy')
info_paddle = diff.load_info('../log_reprod/lr_paddle.npy')
diff.compare_info(info1=info_torch, info2=info_paddle)
diff.report(diff_method='mean', diff_threshold=1e-5, path='../log_diff/loss_diff.txt')











