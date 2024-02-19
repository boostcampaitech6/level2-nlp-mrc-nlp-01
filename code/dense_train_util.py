import os
import random
import datasets
import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import pickle
import re
from datasets import load_dataset
from utils_qa import *
from konlpy.tag import Mecab 
from rank_bm25 import BM25Okapi

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class InBatchNegativeRandomDataset(Dataset):
    def __init__(
        self,
        data_name: str,
        max_context_seq_length: int,
        max_question_seq_length: int,
        tokenizer,
        # sample_n:int
    ):  
        preprocess_data = self.preprocess_pos_neg(
            data_name,
            max_context_seq_length,
            max_question_seq_length,
            tokenizer,
            # sample_n
        )

        self.p_input_ids = preprocess_data[0]
        self.p_attension_mask = preprocess_data[1]
        self.p_token_type_ids = preprocess_data[2]

        self.q_input_ids = preprocess_data[3]
        self.q_attension_mask = preprocess_data[4]
        self.q_token_type_ids = preprocess_data[5]

        self.np_input_ids = preprocess_data[6]
        self.np_attension_mask = preprocess_data[7]
        self.np_token_type_ids = preprocess_data[8]

    def __len__(self):
        return self.p_input_ids.size()[0]

    def __getitem__(self, index):
        return (
            self.p_input_ids[index],
            self.p_attension_mask[index],
            self.p_token_type_ids[index],
            self.q_input_ids[index],
            self.q_attension_mask[index],
            self.q_token_type_ids[index],
            self.np_input_ids[index],
            self.np_attension_mask[index],
            self.np_token_type_ids[index]
        )

    def preprocess_pos_neg(
        self,
        data_name: str,
        max_context_seq_length: int,
        max_question_seq_length: int,
        tokenizer,
        # sample_n,
    ):
        data_path = os.path.join('../data', f"{data_name}.json")
        if not os.path.exists(data_path):
            get_negative_dataset(data_name)
        print('Load Dataset...')
        dataset =  load_dataset('json', data_files= data_path)
        print(f"Train length: {len(dataset['train'])}")
        # sample_idx = np.random.choice(range(len(dataset['train'])), sample_n)
        # dataset = dataset['train'][sample_idx]
        dataset = dataset['train']

        q_seqs = tokenizer(
            dataset['question'],
            max_length=max_question_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        p_seqs = tokenizer(
            dataset['context'],
            max_length=max_context_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        np_seqs = tokenizer(
            dataset['hard_negative_context'],
            max_length=max_context_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )


        return (
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
            np_seqs["input_ids"],
            np_seqs["attention_mask"],
            np_seqs["token_type_ids"]
        )
    
def get_hard_negative(question, exclude_context, contexts, mecab, bm25):
    tokenized_question = mecab.morphs(question)
    scores = bm25.get_scores(tokenized_question)
    ranked_contexts = np.argsort(scores)[::-1]  # 점수에 따라 내림차순 정렬
    
    select_idx = random.randrange(0,5)
    set_ranked_contexts = []
    for idx in ranked_contexts[:10]:
        if contexts[idx] != exclude_context:
            set_ranked_contexts.append(contexts[idx])
    return set_ranked_contexts[select_idx]

def get_negative_dataset(save_name):
    mecab = Mecab() 
    dataset = load_dataset('squad_kor_v1')
    contexts = list(
            dict.fromkeys([v["context"] for v in dataset['train']])
        )

    tokenized_contexts = []

    for text in tqdm(contexts, total = len(contexts), desc = 'Tokenizing...'):
        tokenized_contexts.append(mecab.morphs(text))
    print('BM25 Creating..')
    bm25 = BM25Okapi(tokenized_contexts)

    enhanced_dataset = []
    for item in tqdm(dataset['train'], total = len(dataset['train']), desc = 'Datasets Create...'):
        hard_negative_context = get_hard_negative(item['question'], item['context'],contexts, mecab, bm25)
        enhanced_dataset.append({
            'id': item['id'],
            'title': item['title'],
            'context': item['context'],  # 원래 context
            'hard_negative_context': hard_negative_context,  # 추가된 하드 네거티브 context
            'question': item['question'],
            'answers': item['answers']
    })
    save_path = os.path.join('../data/', f"{save_name}.json")
    print('Dataset Save...')
    with open(save_path, 'w') as file:
        json.dump(enhanced_dataset, file)
    print('Process Done!...')