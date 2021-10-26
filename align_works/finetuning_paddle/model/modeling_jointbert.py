# import torch
# import torch.nn as nn
# from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
# from torchcrf import CRF

import paddle
import paddle.nn as nn
from paddlenlp.transformers import BertModel, BertPretrainedModel
from .module import IntentClassifier, SlotClassifier
from .paddle_crf import CRF



class JointBERT(BertPretrainedModel):
    def __init__(self, bert, args, intent_label_lst, slot_label_lst):
        super(JointBERT, self).__init__()
        self.args = args
        self.bert = bert  # Load pretrained bert
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.hidden_size = 768
        self.intent_classifier = IntentClassifier(self.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(self.hidden_size, self.num_slot_labels, args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.bert(input_ids, attention_mask=paddle.unsqueeze(attention_mask, axis=[1, 2]),
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        return sequence_output, pooled_output

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.reshape([-1], intent_label_ids.shape[-1]))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.reshape([-1, self.num_intent_labels]), intent_label_ids.reshape([-1]))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = (attention_mask.reshape([-1]) == 1).astype(paddle.int32)
                    active_logits = slot_logits.reshape([-1, self.num_slot_labels])[active_loss]
                    active_labels = slot_labels_ids.reshape([-1])[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.reshape([-1, self.num_slot_labels]), slot_labels_ids.reshape([-1]))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
