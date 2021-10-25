# JonitBERT-paddle
paddlepaddle复现JointBERT



- 复杂切片赋值问题：三维tensor切片不能直接完成`score += emissions[0, torch.arange(batch_size), tags[0]]`，二三维都是tensor，不能实现
- 实现了，但可能速度会变慢



| Snips(Joint BERT) | Intent | Slot | Sent | ATIS(Joint BERT+CRF) | Intent | Slot  | Sent  |
| ----------------- | ------ | ---- | ---- | -------------------- | ------ | ----- | ----- |
| 论文效果          | 98.6   | 97.0 | 92.8 |                      | 97.9   | 96.0  | 88.6  |
| pytorch代码效果   |        |      |      |                      | 97.54  | 95.90 | 87.91 |
| Ours-paddle       |        |      |      |                      |        |       |       |





