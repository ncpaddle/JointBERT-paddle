import numpy as np
import pickle

# data = np.load('log_reprod/paddle_back.npy', allow_pickle=True)
torch_data = np.load('log_reprod/torch_back.npy', allow_pickle=True).tolist()
paddle_data = np.load('log_reprod/paddle_back.npy', allow_pickle=True).tolist()

new_data_torch = {'loss_back': torch_data['loss_back'][:6]}
new_data_paddle = {'loss_back': paddle_data['loss_back'][:6]}
np.save('log_reprod/torch_back_loss.npy', new_data_torch)
np.save('log_reprod/paddle_back_loss.npy', new_data_paddle)
# print(data['paddle_back_loss'])
# print(data)
# np.save('log_reprod/paddle_back.npy', data)