import paddle
import paddle.nn as nn
from paddlenlp.transformers import BertModel, BertPretrainedModel
from .module import IntentClassifier, SlotClassifier



class JointAlbert(BertPretrainedModel):
    def __init__(self):
        pass