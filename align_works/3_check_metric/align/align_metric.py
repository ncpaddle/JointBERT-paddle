import numpy as np
import paddle
import torch
from reprod_log import ReprodLogger, ReprodDiffHelper


diff = ReprodDiffHelper()
info_torch = diff.load_info('../log_reprod/metric_torch.npy')
info_paddle = diff.load_info('../log_reprod/metric_paddle.npy')
diff.compare_info(info1=info_torch, info2=info_paddle)
diff.report(diff_method='mean', diff_threshold=1e-5, path='../log_diff/metric_diff_log.txt')
