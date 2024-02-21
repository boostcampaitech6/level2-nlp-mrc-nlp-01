import json
import os
import pickle
import time
import random
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi, BM25Plus
from soynlp.word import WordExtractor
from soynlp.tokenizer import MaxScoreTokenizer
from tqdm.auto import tqdm

seed = 2024
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정



@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        # self.indexer = None  # build_faiss()로 생성합니다.
        self.bm25plus = None
        self.tokenize_fn = tokenize_fn
        self.tfidfv = TfidfVectorizer(
            tokenizer=self.tokenize_fn, ngram_range=(1, 3), max_features=50000,
        )
        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        # self.indexer = None  # build_faiss()로 생성합니다.

    def get_sparse_embedding(self) -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        bm25_name = f"bm25plus.bin"
        bm25_path = os.path.join(self.data_path, bm25_name)
        
        pickle_name = f"sparse_embedding.bin"
        tfidfv_name = f"tfidv.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

    
        if os.path.isfile(emd_path) and os.path.isfile(bm25_path):
            with open(bm25_path, "rb") as file:
                self.bm25plus = pickle.load(file)
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
                
            tokenized_corpus = [self.tokenize_fn(doc) for doc in self.contexts]
            self.bm25plus = BM25Plus(tokenized_corpus)
            
            with open(bm25_path, "wb") as file:
                pickle.dump(self.bm25plus, file)
            print("Embedding pickle saved.")

    
    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        assert self.bm25plus is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."
        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )

            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                        # [self.contexts[pid] for pid_list in doc_indices[idx] for pid in pid_list]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            cqas.to_csv("top_k.csv",index=False)
            return cqas


    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        score_path = os.path.join(self.data_path, "BM25_score.bin")
        indice_path = os.path.join(self.data_path, "BM25_indice.bin")
        
        # Pickle 파일 존재 시에 불러오기
        if os.path.isfile(score_path) and os.path.isfile(indice_path):
            with open(score_path, "rb") as file:
                doc_scores = pickle.load(file)
            with open(indice_path, "rb") as file:
                doc_indices = pickle.load(file)
            print("Load BM25 pickle")
            
        else:
            print('Build BM25 pickle')
            doc_scores = []
            doc_indices = []
            for query in tqdm(queries):
                print(f"query : {query}")
                query_tfdif_vec = self.tfidfv.transform([query])
                result_tfidf = query_tfdif_vec * self.p_embedding.T
                # result_tfidf = np.array(result_tfidf)*1000
                # print(f"TFIDF_SCORE : {result_tfidf}")
                dense_result_tfidf = result_tfidf.toarray()
                flat_result_tfidf = dense_result_tfidf.flatten()
                # print(f"TFIDF_Flatten : {flat_result_tfidf}")
                normalized_array1 = (flat_result_tfidf - np.min(flat_result_tfidf)) / (np.max(flat_result_tfidf) - np.min(flat_result_tfidf))

                tokenized_query = self.tokenize_fn(query)

                # print(f'tokenized_query : {tokenized_query}')
                query_scores = self.bm25plus.get_scores(tokenized_query)
                normalized_array2 = (query_scores - np.min(query_scores)) / (np.max(query_scores) - np.min(query_scores))
                print(f'normalized_array1 : {normalized_array1}')
                
                print(f"normalized_array2 : {normalized_array2}")
    
                result = (normalized_array1*0.6) + (normalized_array2*0.4)
                

                sorted_score = np.sort(result)[::-1]
                sorted_id = np.argsort(result)[::-1]

                doc_scores.append(sorted_score[:k])
                doc_indices.append(sorted_id[:k])
            
                
            assert (
                np.sum(doc_scores) != 0
            ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

            with open(score_path, "wb") as f:
                pickle.dump(doc_scores,f)
            with open(indice_path, "wb") as f:
                pickle.dump(doc_indices,f)
            print("Load BM25 pickle")
            
        return doc_scores, doc_indices
