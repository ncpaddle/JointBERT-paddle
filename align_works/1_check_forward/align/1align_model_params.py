import numpy as np
import torch
import paddle


paddle_p = paddle.load('../../atis_params/model_state.pdparams')
torch_p = torch.load('../../atis_params/pytorch_model.bin', map_location=torch.device('cpu'))

# for k, v in paddle_p.items():
#     if 'bert' not in k:
#         if 'linear' in k or 'dense' in k or 'proj' in k:
#             a = v.t().numpy().tolist()
#         else:
#             a = v.numpy().tolist()
#         b = torch_p[k].detach().numpy().tolist()
#         assert a == b, "{} error".format(k)

torch_id = [k for k, v in torch_p.items()]
torch_lst = [v.numpy().tolist() for k, v in torch_p.items()]
paddle_id = [k for k, v in paddle_p.items()]
paddle_lst = []

i = 0
for k, v in paddle_p.items():
    i += 1
    if 'linear' in k or 'dense' in k or 'proj' in k:
        paddle_lst.append(v.t().detach().numpy().tolist())
    else:
        paddle_lst.append(v.detach().numpy().tolist())




for i in range(len(paddle_lst)):
    assert paddle_lst[i] == torch_lst[i], "{}: padde: {}, torch: {} error".format(i, paddle_id[i], torch_id[i])




print('success')