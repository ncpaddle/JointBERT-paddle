
import argparse
import random
from reprod_log import ReprodLogger, ReprodDiffHelper
from finetuning_paddle.model.modeling_jointbert import  JointBERT as PaddleJointBERT
from finetuning_torch.model.modeling_jointbert import JointBERT as TorchJointBERT
import numpy as np
import torch
import pickle
import paddle
import random
from reprod_log import ReprodLogger, ReprodDiffHelper


def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default='data_atis', type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default='snips_model', type=str,
                        help="Path to save, load model")
    parser.add_argument("--data_dir", default="../../data", type=str, help="The input data dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")

    parser.add_argument("--model_type", default="bert", type=str,)

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
    parser.add_argument("--model_name_or_path", default="../../atis_params", type=str, help="")
    args = parser.parse_args()

    return args


args = getArgs()


# 输入fake_data，得到预测时的数据    在pytorch源代码中生成
inputs = pickle.load(open('../../fake_data.bin', 'rb'))
# for k, v in inputs.items():
#     print(k, v.shape)
# input_ids (32, 50)
# attention_mask (32, 50)
# intent_label_ids (32, 1)
# slot_labels_ids (32, 50)
# token_type_ids (32, 50)

# 生成paddle和torch的输入
paddle_inputs = {k: paddle.to_tensor(v) for (k, v) in inputs.items()}
torch_inputs = {k: torch.tensor(v) for (k, v) in inputs.items()}

intent_labels =[
    label.strip() for label in open(
        '../../data_atis/intent_label.txt', 'r', encoding='utf-8')]
slot_labels = [
    label.strip() for label in open(
        '../../data_atis/slot_label.txt', 'r', encoding='utf-8')]


# 加载paddle和torch的模型，得到evaluate的输出
#TODO 改模型代码，在outputs=self.bert()之后接着return，以得到预训练模型的输出
model_paddle = PaddleJointBERT.from_pretrained(args.model_name_or_path,
                                               args=args,
                                               intent_label_lst=intent_labels,
                                               slot_label_lst=slot_labels)
model_paddle.eval()
out_paddle = model_paddle(**paddle_inputs)
print(out_paddle[0].shape)
print(out_paddle[1].shape)


model_torch = TorchJointBERT.from_pretrained(args.model_name_or_path,
                                             args=args,
                                             intent_label_lst=intent_labels,
                                             slot_label_lst=slot_labels)
model_torch.eval()
out_torch = model_torch(**torch_inputs)
print(out_torch[0].shape)
print(out_torch[1].shape)


# 加载ReprodLogger
rl_torch = ReprodLogger()
rl_paddle = ReprodLogger()
rl_torch.add('sequence_output', out_torch[0].detach().numpy())
rl_paddle.add('sequence_output', out_paddle[0].numpy())
rl_torch.add('pooled_output', out_torch[1].detach().numpy())
rl_paddle.add('pooled_output', out_paddle[1].numpy())
rl_torch.save('../log_reprod/pretrainedModel_output_torch.npy')
rl_paddle.save('../log_reprod/pretrainedModel_output_paddle.npy')




diff = ReprodDiffHelper()
info_torch = diff.load_info('../log_reprod/pretrainedModel_output_torch.npy')
info_paddle = diff.load_info('../log_reprod/pretrainedModel_output_paddle.npy')
diff.compare_info(info1=info_torch, info2=info_paddle)
diff.report(diff_method='mean', diff_threshold=1e-5, path='../log_diff/pretrainedModel_diff.txt')














