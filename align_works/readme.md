# 对齐工作说明

## 1. 模型结构对齐
见文件夹`1_check_forward`

### 预训练模型精度

| sequence_output       | pooled_output          |
| --------------------- | ---------------------- |
| 4.876189905189676e-07 | 3.2371076486015227e-07 |

### 整体模型精度

| intent_logits          | slot_logits           | total_loss |
| ---------------------- | --------------------- | ---------- |
| 1.3917054957346409e-06 | 5.844579504810099e-07 | 0.0        |

## 2. 验证/测试集数据读取对齐

见文件夹`2_check_devdata_testdata`

```
[2021/10/26 17:39:44] root INFO: test_0_input_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_0_attention_mask: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_0_intent_label_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_0_slot_labels_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_0_token_type_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_1_input_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_1_attention_mask: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_1_intent_label_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_1_slot_labels_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_1_token_type_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_2_input_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_2_attention_mask: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_2_intent_label_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_2_slot_labels_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_2_token_type_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_3_input_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_3_attention_mask: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_3_intent_label_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_3_slot_labels_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_3_token_type_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_4_input_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_4_attention_mask: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_4_intent_label_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_4_slot_labels_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: test_4_token_type_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_0_input_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_0_attention_mask: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_0_intent_label_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_0_slot_labels_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_0_token_type_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_1_input_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_1_attention_mask: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_1_intent_label_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_1_slot_labels_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_1_token_type_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_2_input_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_2_attention_mask: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_2_intent_label_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_2_slot_labels_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_2_token_type_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_3_input_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_3_attention_mask: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_3_intent_label_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_3_slot_labels_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_3_token_type_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_4_input_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_4_attention_mask: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_4_intent_label_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_4_slot_labels_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: dev_4_token_type_ids: 
[2021/10/26 17:39:44] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 17:39:44] root INFO: diff check passed

```

## 3. 评估指标和前向损失对齐

见文件夹`3_check_metric_loss`

```
[2021/10/26 20:40:15] root INFO: loss: 
[2021/10/26 20:40:15] root INFO: 	mean diff: check passed: True, value: 7.737960132647714e-07
[2021/10/26 20:40:15] root INFO: intent_acc: 
[2021/10/26 20:40:15] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 20:40:15] root INFO: slot_precision: 
[2021/10/26 20:40:15] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 20:40:15] root INFO: slot_recall: 
[2021/10/26 20:40:15] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 20:40:15] root INFO: slot_f1: 
[2021/10/26 20:40:15] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 20:40:15] root INFO: sementic_frame_acc: 
[2021/10/26 20:40:15] root INFO: 	mean diff: check passed: True, value: 0.0
[2021/10/26 20:40:15] root INFO: diff check passed
```

## 4. 优化器对齐

见7. 反向对齐

## 5. 学习率对齐

见文件夹`5_check_learning_rate`

| 学习率对齐精度 |
| -------------- |
| 0.0            |

## 6. 正则化策略对齐

见7. 反向对齐

## 7. 反向对齐

见文件夹`4-6-7_check_optim-norm-backward`

| 反向对齐精度          |
| --------------------- |
| 8.050096221268177e-05 |



## 8. 训练集数据读取对齐

见文件夹`8_check_forward_data`

| 训练集数据对齐精度 |
| ------------------ |
| 0.0                |

## 9. 网络初始化对齐

> 使用pytorch的网络初始化权重来初始化paddle的网络并进行训练。



## 10. 模型训练对齐

见实验结果
