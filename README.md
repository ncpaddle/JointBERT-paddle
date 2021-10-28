# JointBERT-paddle

(Unofficial) **Paddle** implementation of `JointBERT`: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)

## Model Architecture

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68875755-b2f92900-0746-11ea-8819-401d60e4185f.png" />  
</p>

- Predict `intent` and `slot` at the same time from **one BERT model** (=Joint model)
- total_loss = intent_loss + coef \* slot_loss (Change coef with `--slot_loss_coef` option)
- **If you want to use CRF layer, give `--use_crf` option**

## Dependencies

- python>=3.6
- paddle == 2.1.3
- paddlenlp == 2.0.0

  

## Dataset

|       | Train  | Dev | Test | Intent Labels | Slot Labels |
| ----- | ------ | --- | ---- | ------------- | ----------- |
| ATIS  | 4,478  | 500 | 893  | 21            | 120         |
| Snips | 13,084 | 700 | 700  | 7             | 72          |

- The number of labels are based on the _train_ dataset.
- Add `UNK` for labels (For intent and slot labels which are only shown in _dev_ and _test_ dataset)
- Add `PAD` for slot label

## Training & Evaluation

```bash
# Train
$ python main.py --task {task_name} \
                  --model_type {model_type} \
                  --model_dir {model_dir_name} \
                  --do_train --do_eval \
                  --use_crf

# Train For ATIS
$ python main.py --task atis \
                --model_type bert \
                --model_dir atis_model \
                --do_train --do_eval \
                --use_crf \
                --data_dir ../data \
                --model_name_or_path bert-base-uncased \
                --num_train_epochs 30.0
# Train For Snips
$ python main.py --task snips \
                --model_type bert \
                --model_dir snips_model \
                --do_train --do_eval \
                --data_dir ../data \
                --model_name_or_path bert-base-uncased \
                --num_train_epochs 30.0
              
              
# Evaluation For ATIS
$ python main.py --task atis \
                --model_type bert \
                --model_dir atis_model \
                --do_eval \
                --use_crf \
                --data_dir ../data \
                --model_name_or_path bert-base-uncased \
                --num_train_epochs 30.0

# Evaluation For Snips
$ python main.py --task snips \
                --model_type bert \
                --model_dir snips_model \
                --do_eval \
                --data_dir ../data \
                --model_name_or_path bert-base-uncased \
                --num_train_epochs 30.0
                
                
```

## 

## Results

- Run 5 ~ 10 epochs (Record the best result)
- Only test with `uncased` model
- ALBERT xxlarge sometimes can't converge well for slot prediction.

|           |                      | Intent acc (%) | Slot F1 (%) | Sentence acc (%) |
| --------- | -------------------- | -------------- | ----------- | ---------------- |
| **Snips** | BERT (paper)         | 98.6           | 97.0        | 92.8             |
|           | BERT (pytorch)       | 98.7           | 96.1        | 91.1             |
|           | BERT (paddle)        | 98.6           | 96.1        | 91.4             |
|           |                      |                |             |                  |
|           | BERT + CRF (paper)   | 98.4           | 96.7        | 92.6             |
|           | BERT + CRF (pytorch) |                |             |                  |
|           | BERT + CRF (paddle)  | 98.5           | 96.8        | 92.7             |
|           |                      |                |             |                  |
| **ATIS**  | BERT (paper)         | 97.5           | 96.1        | 88.2             |
|           | BERT (pytorch)       |                |             |                  |
|           | BERT (paddle)        | 97.5           | 95.6        | 87.6             |
|           |                      |                |             |                  |
|           | BERT + CRF (paper)   | 97.9           | 96.0        | 88.6             |
|           | BERT + CRF (pytorch) | 97.6           | 95.9        | 88.7             |
|           | BERT + CRF (paddle)  | 97.4           | 95.8        | 88.0             |

## Align

- `forward_diff`: 
- `metric_diff`:
- `loss_diff`:
- `backward_diff`:
- `train_align`:

More details about align works in here.











第五十篇RemBERT，没有代码，只有模型，怎么做对齐
