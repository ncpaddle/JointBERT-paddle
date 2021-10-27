import numpy as np
import torch
import paddle
from reprod_log import ReprodLogger, ReprodDiffHelper

diff = ReprodDiffHelper()
info_torch = diff.load_info('log_reprod/torch_back_loss.npy')
info_paddle = diff.load_info('log_reprod/paddle_back_loss.npy')

# info_paddle['loss_back'] = info_paddle['loss_back'][:6]
# info_torch['loss_back'] = info_torch['loss_back'][:6]

diff.compare_info(info1=info_torch, info2=info_paddle)
diff.report(diff_method='mean', diff_threshold=1e-5, path='log_diff/loss_diff.txt')











