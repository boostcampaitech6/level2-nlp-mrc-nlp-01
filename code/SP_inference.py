"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""
from single_passage import post_process_voting

import argparse
import logging
import sys, os
from typing import Callable, Dict, List, NoReturn, Tuple
import pandas as pd 

import numpy as np
from arguments import DataTrainingArguments, ModelArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
    load_metric,
)
from konlpy.tag import Mecab
from rank_bm25 import BM25Okapi
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils_qa import *
from retrieval import SparseRetrieval, DenseRetrieval, HybridRetrieval
from dense_models import BertEncoder, RobertaEncoder
logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.do_train = True
    training_args.output_dir = '../outputs/' + model_args.model_name_or_path.split('/')[-1]
    training_args.save_total_limit = 1
    if not os.path.exists(training_args.output_dir):
        os.mkdir(training_args.output_dir)

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )
    # mecab = Mecab()
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )
    torch.cuda.empty_cache()
    if data_args.dense_encoder_type == 'dense':
        print('Retriever Part Start...')
        if data_args.eval_retrieval:
            if data_args.single_passage:
                datasets, doc_scores = run_dense_retrieval(
                    tokenizer.tokenize, datasets, training_args, data_args,
                )
            else:
                datasets = run_dense_retrieval(
                    tokenizer.tokenize, datasets, training_args, data_args,
        )
        else:
            datasets = run_dense_retrieval(
                tokenizer.tokenize, datasets, training_args, data_args,
        )
        print('Retriever Part End...')
        torch.cuda.empty_cache()
        # eval or predict mrc model
        if training_args.do_eval or training_args.do_predict:
            run_mrc(data_args, training_args, model_args, datasets, tokenizer, model, doc_scores)
        torch.cuda.empty_cache()

    elif data_args.dense_encoder_type == 'sparse':
        if data_args.eval_retrieval:
            if data_args.single_passage:
                datasets, doc_scores = run_sparse_retrieval(
                    tokenizer.tokenize, datasets, training_args, data_args,
                )
            else:
                datasets = run_sparse_retrieval(
                    tokenizer.tokenize, datasets, training_args, data_args,
        )
        torch.cuda.empty_cache()
        # eval or predict mrc model
        if training_args.do_eval or training_args.do_predict:
            run_mrc(data_args, training_args, model_args, datasets, tokenizer, model, doc_scores)

    elif data_args.dense_encoder_type == 'hybrid':
        print('Retriever Part Start...')
        if data_args.eval_retrieval:
            if data_args.single_passage:
                datasets, doc_scores = run_hybrid_retrieval(
                    tokenizer.tokenize, datasets, training_args, data_args,
                )
            else:
                datasets = run_hybrid_retrieval(
                    tokenizer.tokenize, datasets, training_args, data_args,
        )
        print('Retriever Part End...')
        torch.cuda.empty_cache()
        # eval or predict mrc model
        if training_args.do_eval or training_args.do_predict:
            run_mrc(data_args, training_args, model_args, datasets, tokenizer, model, doc_scores)
        torch.cuda.empty_cache()

def run_sparse_retrieval(
    # tokenize_fn: Callable[[str], List[str]],
    args: ModelArguments,
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(
        args=ModelArguments, data_path=data_path, context_path=context_path, remove_char=data_args.remove_char
    )
    retriever.get_sparse_embedding()

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        if data_args.single_passage:
            doc_scores, df_list = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval, single_passage=True)
        else:
            df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval, single_passage=False)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    if data_args.single_passage:
        datasets_list = []
        for i in range(data_args.top_k_retrieval):
            dataset = DatasetDict({"validation": Dataset.from_pandas(df_list[i], features=f)})
            datasets_list.append(dataset)
        return datasets_list, doc_scores
    else:
        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        return datasets

def run_dense_retrieval(
    args: ModelArguments,
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    p_encoder_path: str = '../dense_model/p_encoder.pt',
    q_encoder_path: str = '../dense_model/q_encoder.pt',
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = DenseRetrieval(
        args = ModelArguments,
        p_encoder_path = p_encoder_path,
        q_encoder_path = q_encoder_path,
        data_path=data_path, 
        context_path=context_path
    )
    retriever.get_dense_embedding()

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        if data_args.single_passage:
            doc_scores, df_list = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval, single_passage=True)
        else:
            df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval, single_passage=False)


    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    if data_args.single_passage:
        datasets_list = []
        for i in range(data_args.top_k_retrieval):
            dataset = DatasetDict({"validation": Dataset.from_pandas(df_list[i], features=f)})
            datasets_list.append(dataset)
        return datasets_list, doc_scores
    else:
        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        return datasets

def run_hybrid_retrieval(
    # tokenize_fn: Callable[[str], List[str]],
    args: ModelArguments,
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    p_encoder_path: str = '../dense_model/p_encoder.pt',
    q_encoder_path: str = '../dense_model/q_encoder.pt',
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = HybridRetrieval(
        args = ModelArguments, 
        p_encoder_path = p_encoder_path, 
        q_encoder_path = q_encoder_path,
        data_path=data_path, 
        context_path=context_path
    )
    # retriever.get_sparse_embedding()

    # if data_args.use_faiss:
    #     retriever.build_faiss(num_clusters=data_args.num_clusters)
    #     df = retriever.retrieve_faiss(
    #         datasets["validation"], topk=data_args.top_k_retrieval
    #     )
    # else:
    if data_args.single_passage:
        doc_scores, df_list = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval, single_passage=True)
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval, single_passage=False)
    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    if data_args.single_passage:
        datasets_list = []
        for i in range(data_args.top_k_retrieval):
            dataset = DatasetDict({"validation": Dataset.from_pandas(df_list[i], features=f)})
            datasets_list.append(dataset)
        return datasets_list, doc_scores
    else:
        datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        return datasets

def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
    doc_scores=None,
) -> NoReturn:

    if data_args.single_passage:
        k = data_args.top_k_retrieval
    else:
        k = 1
        datasets = [datasets]
    eval_dataset = []

    for i in tqdm(range(k)):
        # eval 혹은 prediction에서만 사용함
        column_names = datasets[i]["validation"].column_names

        question_column_name = "question" if "question" in column_names else column_names[0]
        context_column_name = "context" if "context" in column_names else column_names[1]
        answer_column_name = "answers" if "answers" in column_names else column_names[2]

        # Padding에 대한 옵션을 설정합니다.
        # (question|context) 혹은 (context|question)로 세팅 가능합니다.
        pad_on_right = tokenizer.padding_side == "right"

        # 오류가 있는지 확인합니다.
        last_checkpoint, max_seq_length = check_no_error(
            data_args, training_args, datasets[i], tokenizer
        )
        
        eval_dataset.append(datasets[i]["validation"])

        # Validation preprocessing / 전처리를 진행합니다.
        def prepare_validation_features(examples):
            # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
            # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=data_args.doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
                padding="max_length" if data_args.pad_to_max_length else False,
            )

            # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

            # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
            # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
            tokenized_examples["example_id"] = []

            for j in range(len(tokenized_examples["input_ids"])):
                # sequence id를 설정합니다 (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(j)
                context_index = 1 if pad_on_right else 0

                # 하나의 example이 여러개의 span을 가질 수 있습니다.
                sample_index = sample_mapping[j]
                tokenized_examples["example_id"].append(examples["id"][sample_index])

                # context의 일부가 아닌 offset_mapping을 None으로 설정하여 토큰 위치가 컨텍스트의 일부인지 여부를 쉽게 판별할 수 있습니다.
                tokenized_examples["offset_mapping"][j] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][j])
                ]
            return tokenized_examples

        # Validation Feature 생성
        eval_dataset[i] = eval_dataset[i].map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        # Data collator
        # flag가 True이면 이미 max length로 padding된 상태입니다.
        # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )

        # Post-processing:
        def post_processing_function(
            examples,
            features,
            predictions: Tuple[np.ndarray, np.ndarray],
            training_args: TrainingArguments,
        ) -> EvalPrediction:
            # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
            predictions = postprocess_qa_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
                max_answer_length=data_args.max_answer_length,
                output_dir=training_args.output_dir, 
                file_name=training_args.output_dir.split('/')[-1]
            )
            # Metric을 구할 수 있도록 Format을 맞춰줍니다.
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()
            ]

            if training_args.do_predict:
                return formatted_predictions
            elif training_args.do_eval:
                references = [
                    {"id": ex["id"], "answers": ex[answer_column_name]}
                    for ex in datasets[i]["validation"]
                ]

                return EvalPrediction(
                    predictions=formatted_predictions, label_ids=references
                )

        metric = load_metric("squad")

        def compute_metrics(p: EvalPrediction) -> Dict:
            return metric.compute(predictions=p.predictions, references=p.label_ids)

        print("init trainer...")
        # Trainer 초기화
        if data_args.single_passage:
            training_args.output_dir = '../outputs/' + model_args.model_name_or_path.split('/')[-1] + f'/split_prediction/{i}_pred'
        trainer = QuestionAnsweringTrainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=eval_dataset[i],
            eval_examples=datasets[i]["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=post_processing_function,
            compute_metrics=compute_metrics,
        )

        logger.info("*** Evaluate ***")

        #### eval dataset & eval example - predictions.json 생성됨
        if training_args.do_predict:
            predictions = trainer.predict(
                test_dataset=eval_dataset[i], test_examples=datasets[i]["validation"]
            )

            # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
            print(
                "No metric can be presented because there is no correct answer given. Job done!"
            )

        if training_args.do_eval:
            metrics = trainer.evaluate()
            metrics["eval_samples"] = len(eval_dataset)

            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)

    if data_args.single_passage:
        post_process_voting(doc_scores, training_args.output_dir, data_args.top_k_retrieval, pd.DataFrame(load_from_disk(data_args.dataset_name)['validation']))
                

if __name__ == "__main__":
    main()

