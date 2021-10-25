import paddle
import torch

x = torch.tensor(
    [   [[1 , 4 , 3 , 8 ],
         [4 , 5 , 6 , 0 ],
         [7 , 4 , 0 , 5 ]],
        [[12, 6 , 38, 0 ],
         [42, 7 , 9 , 1 ],
         [79, 40, 20, 45]]])


a = [1, 1]
b = [1, 2]

index = []
for i in range(len(a)):
    index.append([0, a[i], b[i]])

print(index)

print(paddle.gather_nd(x, paddle.to_tensor(index)))