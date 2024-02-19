import json 
from tqdm import tqdm, trange
import random
import argparse
import os

from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from transformers import (AutoTokenizer,
                          AutoConfig,
                          BertModel, 
                          BertPreTrainedModel, 
                          AdamW, 
                          TrainingArguments, 
                          RobertaModel, 
                          RobertaPreTrainedModel,
                          get_linear_schedule_with_warmup)
from dense_models import RobertaEncoder, BertEncoder
from utils_qa import get_negative_dataset
from dense_train_util import InBatchNegativeRandomDataset


def compute_loss(q_outputs, p_outputs, hn_outputs, margin=0.5):
    """
    Triplet loss 계산을 위한 함수
    - q_outputs: 질문 인코딩 벡터
    - p_outputs: positive context 인코딩 벡터
    - hn_outputs: hard negative context 인코딩 벡터
    - margin: 마진 값, positive context와 hard negative context 간의 최소 거리 차이
    """
    # 질문과 긍정적 컨텍스트 간의 유사도 점수 계산
    positive_scores = torch.diag(torch.matmul(q_outputs, p_outputs.T))
    
    # 질문과 하드 네거티브 컨텍스트 간의 유사도 점수 계산
    negative_scores = torch.diag(torch.matmul(q_outputs, hn_outputs.T))
    
    # Triplet loss 계산
    # losses = F.relu(positive_scores - negative_scores + margin)
    losses = F.relu(negative_scores - positive_scores + margin)
    loss = losses.mean()
    
    return loss

def train(args, dataset, p_model, q_model):
    # 변경점
    # answer 하나 라벨 별 균등
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler= train_sampler, batch_size = args.per_device_train_batch_size)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in p_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in p_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in q_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in q_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    global_step = 0

    p_model.zero_grad()
    q_model.zero_grad()
    torch.cuda.empty_cache()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            p_model.train()
            q_model.train()
            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)
            neg_batch_ids = []
            neg_batch_att = []
            neg_batch_tti = []

            for batch_in_sample_idx in range(args.per_device_train_batch_size):
                neg_batch_ids.append(
                    batch[6][:][batch_in_sample_idx].unsqueeze(0)
                )
                neg_batch_att.append(
                    batch[7][:][batch_in_sample_idx].unsqueeze(0)
                )
                neg_batch_tti.append(
                    batch[8][:][batch_in_sample_idx].unsqueeze(0)
                )
            neg_batch_ids = torch.cat(neg_batch_ids)
            neg_batch_att = torch.cat(neg_batch_att)
            neg_batch_tti = torch.cat(neg_batch_tti)
            p_inputs = {
                "input_ids": torch.cat((batch[0], neg_batch_ids), 0).cuda(),
                "attention_mask": torch.cat((batch[1], neg_batch_att), 0).cuda(),
                "token_type_ids": torch.cat((batch[2], neg_batch_tti), 0).cuda(),
                }
            q_inputs = {
                "input_ids": batch[3].cuda(),
                "attention_mask": batch[4].cuda(),
                "token_type_ids": batch[5].cuda(),
            }

            p_outputs = p_model(**p_inputs)  # (batch_size, emb_dim)
            q_outputs = q_model(**q_inputs)  # (batch_size, emb_dim)
            # hn_outputs = p_model(**hn_inputs) # (batch_size, emb_dim)

            # Calculate similarity score & loss
            # loss = compute_loss(q_outputs, p_outputs, hn_outputs, margin= 0.5)
            sim_scores = torch.matmul(
            q_outputs, torch.transpose(p_outputs, 0, 1)
            )  # (batch_size, emb_dim) x (emb_dim, batch_size * 2) = (batch_size, batch_size * 2)

            # 정답은 대각선의 성분들 -> 0 1 2 ... batch_size - 1
            targets = torch.arange(0, args.per_device_train_batch_size).long()
            if torch.cuda.is_available():
                targets = targets.to("cuda")

            sim_scores = F.log_softmax(sim_scores, dim=1)
            loss = F.nll_loss(sim_scores, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()
            q_model.zero_grad()
            p_model.zero_grad()
            global_step += 1

            torch.cuda.empty_cache()

    return p_model, q_model

def main(args):
    torch.manual_seed(2024)
    torch.cuda.manual_seed(2024)
    np.random.seed(2024)
    random.seed(2024)


    model_checkpoint = args.model_checkpoint

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_dataset = InBatchNegativeRandomDataset(
        data_name = 'hard_negative_datasets',
        max_context_seq_length = args.max_context_seq_length,
        max_question_seq_length = args.max_question_seq_length,
        tokenizer = tokenizer,
        # sample_n = args.sample_n
    )

    p_encoder = RobertaEncoder.from_pretrained(model_checkpoint)
    q_encoder = RobertaEncoder.from_pretrained(model_checkpoint)

    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()

        print('GPU enabled')

    train_args = TrainingArguments(
        output_dir="dense_retireval",
        save_total_limit=1, 
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        num_train_epochs=3,
        weight_decay=0.01)
    
    p_encoder, q_encoder = train(train_args, train_dataset, p_encoder, q_encoder)

    torch.save(p_encoder.state_dict(), '/data/ephemeral/odqa/dense_model/' + 'p_encoder.pt')
    torch.save(q_encoder.state_dict(), '/data/ephemeral/odqa/dense_model/' + 'q_encoder.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name', '-dp', default= 'hard_negative_datasets', type=str
    )
    # parser.add_argument(
    #     '--sample_n', '-sn', default= 8000, type=int
    # )
    parser.add_argument(
        '--model_checkpoint', '-mc', default="klue/roberta-base", type=str
    )
    parser.add_argument(
        '--batch_size', '-bs', default=16, type=int
    )
    parser.add_argument(
        "--max_context_seq_length", type=int, default=512
        )
    parser.add_argument(
        "--max_question_seq_length", type=int, default=64
        )
    args = parser.parse_args()

    main(args)