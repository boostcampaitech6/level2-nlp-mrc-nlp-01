import json 
from tqdm import tqdm, trange
import random
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from transformers import AutoTokenizer, BertModel, BertPreTrainedModel, AdamW, TrainingArguments, RobertaModel, RobertaPreTrainedModel,get_linear_schedule_with_warmup


class RobertaEncoder(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaEncoder, self).__init__(config)

        self.roberta = RobertaModel(config)
        # 변경점
        self.init_weights()

    def forward(self, input_ids, attention_mask = None, token_type_ids = None):
        outputs =self.roberta(input_ids, attention_mask, token_type_ids)
        pooled_output =outputs['pooler_output']
        return pooled_output
    
class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        # 변경점
        self.init_weights()

    def forward(self, input_ids, attention_mask = None, token_type_ids = None):
        outputs =self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output =outputs['pooler_output']
        return pooled_output