import os
from finetuning_paddle.model.modeling_jointbert import JointBERT as paddle_JointBERT
from finetuning_torch.model.modeling_jointbert import JointBERT as torch_JonitBERT
import argparse
from finetuning_paddle.utils import MODEL_CLASSES, compute_metrics, get_intent_labels, get_slot_labels
from transformers import BertConfig
import pickle
import paddle
import torch
import numpy as np
from reprod_log import ReprodLogger, ReprodDiffHelper

intent_label_lst = [label.strip() for label in open("../../../data/atis/intent_label.txt", 'r', encoding='utf-8')]
slot_label_lst = [label.strip() for label in open("../../../data/atis/slot_label.txt", 'r', encoding='utf-8')]

def getparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default='atis', required=False, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default='atis_model', required=False, type=str,
                        help="Path to save, load model")
    parser.add_argument("--data_dir", default="data", type=str, help="The input data dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

    # CRF option
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--slot_pad_label", default="PAD", type=str,
                        help="Pad token for slot label pad (to be ignore when calculate loss)")
    parser.add_argument("--model_name_or_path", default="", type=str, help="")
    args = parser.parse_args()
    return args
args = getparser()
args.model_type='bert'
args.task = 'atis'
args.use_crf = True
config = BertConfig.from_pretrained('../../../atis_params', finetuning_task=args.task)
paddle_model = paddle_JointBERT.from_pretrained('../../../atis_params',
                                                args=args,
                                                intent_label_lst=intent_label_lst,
                                                slot_label_lst=slot_label_lst)

torch_model = torch_JonitBERT.from_pretrained('../../../atis_params',
                                                      config=config,
                                                      args=args,
                                                      intent_label_lst=intent_label_lst,
                                                      slot_label_lst=slot_label_lst)

input_data = inputs = pickle.load(open('../../fake_data.bin', 'rb'))
# del input_data['intent_label_ids']
# del input_data['slot_labels_ids']
# 生成paddle和torch的输入
paddle_inputs = {k: paddle.to_tensor(v) for (k, v) in inputs.items()}
torch_inputs = {k: torch.tensor(v) for (k, v) in inputs.items()}

paddle_model.eval()
torch_model.eval()

paddle_out = paddle_model(**paddle_inputs)
torch_out = torch_model(**torch_inputs)

# 加载ReprodLogger
rl_torch = ReprodLogger()
rl_paddle = ReprodLogger()
rl_torch.add('intent_logits', torch_out[1][0].detach().numpy())
rl_paddle.add('intent_logits', paddle_out[1][0].numpy())
rl_torch.add('slot_logits', torch_out[1][1].detach().numpy())
rl_paddle.add('slot_logits', paddle_out[1][1].numpy())
rl_torch.add('total_loss', torch_out[0].detach().numpy())
rl_paddle.add('total_loss', paddle_out[0].numpy())

rl_torch.save('../log_reprod/Model_output_torch.npy')
rl_paddle.save('../log_reprod/Model_output_paddle.npy')




diff = ReprodDiffHelper()
info_torch = diff.load_info('../log_reprod/Model_output_torch.npy')
info_paddle = diff.load_info('../log_reprod/Model_output_paddle.npy')
diff.compare_info(info1=info_torch, info2=info_paddle)
diff.report(diff_method='mean', diff_threshold=1e-5, path='../log_diff/Model_diff.txt')














