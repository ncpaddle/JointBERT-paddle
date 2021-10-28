# import torch.nn as nn
import paddle.nn as nn

class IntentClassifier(nn.Layer):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels, weight_attr=nn.initializer.KaimingNormal(),
                                bias_attr=nn.initializer.KaimingNormal())

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class SlotClassifier(nn.Layer):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels, weight_attr=nn.initializer.KaimingNormal(),
                                bias_attr=nn.initializer.KaimingNormal())

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)
