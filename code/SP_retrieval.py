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
from tqdm.auto import tqdm
import re

import torch
from transformers import RobertaConfig, AutoConfig, AutoTokenizer
from dense_models import BertEncoder, RobertaEncoder
from rank_bm25 import BM25Okapi
from konlpy.tag import Mecab
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, OkapiBM25Model, LuceneBM25Model
from gensim.similarities import SparseMatrixSimilarity
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
        args,
        # tokenize_fn,
        # tokenizer,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        remove_char=False,
        single_passage=True
    ):

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        # self.data_path = data_path
        tokenizer = AutoTokenizer.from_pretrained(args.config_name)
        self.tokenize_fn = tokenizer.tokenize

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        if remove_char:
            pattern = '[^A-Za-z0-9가-힣 ]'
            self.contexts_before=[x for x in self.contexts]
            for i in range(len(self.contexts)):
                self.contexts[i]=re.sub(pattern, '', self.contexts_before[i])
        
        self.tokenized_corpus = [
            self.tokenize_fn(text) for text in self.contexts
        ]

        #self.sparse_retrieval.get_sparse_embedding()
        #self.p_embedding = None #self.sparse_retrieval.p_embedding  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

    def get_sparse_embedding(self):

        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        #------------------------------------------------------------
        bm_dic_path = os.path.join('../data/', "bm_dic.bin")
        bm_model_path = os.path.join('../data/', "bm_model.bin")
        bm_index_path = os.path.join('../data/',"bm_index.bin")
        
        if os.path.isfile(bm_dic_path) and os.path.isfile(bm_model_path) and os.path.isfile(bm_index_path):
            with open(bm_dic_path, 'rb') as file:
                self.dictionary = pickle.load(file)
            with open(bm_model_path, 'rb') as file:
                self.bm25_model = pickle.load(file)
            with open(bm_index_path, 'rb') as file:
                self.bm25_index = pickle.load(file)
            print("Embedding pickle load.")
        else:
            self.dictionary = Dictionary(self.tokenized_corpus)
            self.bm25_model = OkapiBM25Model(dictionary=self.dictionary)
            # self.bm25_model = LuceneBM25Model(dictionary=self.dictionary)
            self.bm25_corpus = self.bm25_model[list(map(self.dictionary.doc2bow, self.tokenized_corpus))]
            self.bm25_index = SparseMatrixSimilarity(self.bm25_corpus, 
                                                num_docs=len(self.tokenized_corpus), 
                                                num_terms=len(self.dictionary),
                                                normalize_queries=False, 
                                                normalize_documents=False)
            with open(bm_dic_path, 'wb') as file:
                pickle.dump(self.dictionary, file)
            with open(bm_model_path, 'wb') as file:
                pickle.dump(self.bm25_model, file)
            with open(bm_index_path, 'wb') as file:
                pickle.dump(self.bm25_index, file)
            print("Embedding pickle saved.")
 

    def build_faiss(self, num_clusters=64) -> NoReturn:

        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1, single_passage: Optional[bool] = True,
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

        # assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )

            if single_passage:
                doc_scores = np.array(doc_scores)
                doc_scores = doc_scores / np.max(doc_scores)
                cqas_list = [] 
                for i in range(topk):
                    total = []
                    for idx, example in enumerate(
                        tqdm(query_or_dataset, desc="Sparse retrieval: ")
                    ):
                        tmp = {
                            # Query와 해당 id를 반환합니다.
                            "question": example["question"],
                            "id": example["id"],
                            # Retrieve한 Passage의 id, context를 반환합니다.
                            "context": self.contexts[doc_indices[idx][i]],
                        }
                        if "context" in example.keys() and "answers" in example.keys():
                            # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                            tmp["original_context"] = example["context"]
                            tmp["answers"] = example["answers"]
                        total.append(tmp)
                    cqas = pd.DataFrame(total)
                    cqas_list.append(cqas)    
                return doc_scores, cqas_list
            
            else:
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
                        ),
                    }
                    if "context" in example.keys() and "answers" in example.keys():
                        # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                        tmp["original_context"] = example["context"]
                        tmp["answers"] = example["answers"]
                    total.append(tmp)

                cqas = pd.DataFrame(total)
                return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        with timer("transform"):
            query_vec = self.tokenize_fn(query)
        # assert (
        #     np.sum(query_vec) != 0
        # ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            #-----------------------------------
            bm_query = self.bm25_model[self.dictionary.doc2bow(query_vec)]
            doc_score = self.bm25_index[bm_query]
            doc_indices = np.argsort(doc_score)[::-1][:k]
            #-----------------------------------
        #     doc_score = self.p_embedding.get_scores(query_vec)[:k]
        #     doc_indices = np.argsort(doc_score)[::-1]
            
        # doc_score = sorted(doc_score, reverse = True)
        doc_score = sorted(doc_score, reverse = True)[:k]
        return doc_score, doc_indices

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
        #-----------------------------------

        query_vec = [self.tokenize_fn(i) for i in queries]
        doc_scores = []
        doc_indices = []
        for i in tqdm(range(len(query_vec)), total =len(query_vec), desc = 'BM25 get scores...'):
            bm_query = self.bm25_model[self.dictionary.doc2bow(query_vec[i])]
            scores = self.bm25_index[bm_query]
            indices = np.argsort(scores)[::-1][:k]
            scores = sorted(scores, reverse = True)[:k]
            doc_scores.append(scores)
            doc_indices.append(indices)
        #-----------------------------------

        # query_vec = [self.tokenize_fn(i) for i in queries]
        # doc_scores = []
        # doc_indices = []
        # for i in tqdm(range(len(query_vec)), total =len(query_vec), desc = 'BM25 get scores...'):
            # scores = self.p_embedding.get_scores(query_vec[i])[:k]
            # doc_scores.append(sorted(scores, reverse = True))
            # indices = np.argsort(scores)[::-1]
            # doc_indices.append(indices.tolist())

        return doc_scores, doc_indices

    def retrieve_faiss(
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
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
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
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
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

        query_vecs = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()

class DenseRetrieval:
    def __init__(
            self,
            # config,
            args,
            # tokenizer,
            p_encoder_path,
            q_encoder_path,
            data_path: Optional[str] = "../data/",
            context_path: Optional[str] = "wikipedia_documents.json",
            ):

        
        self.data_path = data_path
        self.config = AutoConfig.from_pretrained(args.config_name_dpr)
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Transform by Encoder
        # 추가------------------------------------------------------------------------
        self.p_encoder = RobertaEncoder(self.config)
        self.p_encoder.load_state_dict(torch.load(p_encoder_path))
        
        self.q_encoder = RobertaEncoder(self.config)
        self.q_encoder.load_state_dict(torch.load(q_encoder_path))
        if torch.cuda.is_available():
            self.p_encoder.cuda()
            self.q_encoder.cuda()
        # self.tokenizer = tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.config_name_dpr)
        # 추가------------------------------------------------------------------------

        #self.dense_retreival.get_dense_embedding()
        self.p_embedding = None#self.dense_retreival.p_embedding  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.
    # 변경------------------------------------------------------------------------
    def get_dense_embedding(self):

        """
        Summary:
            Passage Embedding을 만들고
            Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"dense_embedding.npy"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            self.p_embedding = self.p_embedding.to('cuda')
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.passage_embedding(self.contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")
     # 변경------------------------------------------------------------------------
    
    # 추가------------------------------------------------------------------------
    def passage_embedding(self,valid_corpus):
        p_embs = []
        with torch.no_grad():
            self.p_encoder.eval()
            for p in tqdm(valid_corpus, total = len(valid_corpus), desc = "Dense Embedding Create..."):
                inputs = self.tokenizer(p, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                p_emb = self.p_encoder(**inputs).to('cpu').numpy()
                p_embs.append(p_emb)
        torch.cuda.empty_cache()
        p_embs = torch.Tensor(p_embs).squeeze()

        return p_embs
    
    # 추가------------------------------------------------------------------------
    def query_embedding(self, queries):
        with torch.no_grad():
            self.q_encoder.eval()
            if isinstance(queries, str):
                queries = [queries]  # Ensure queries is a list of strings
            q_seqs_val = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
            q_emb = self.q_encoder(**q_seqs_val) #(num_query, emb_dim)
        return q_emb
    
    def build_faiss(self, num_clusters=64) -> NoReturn:

        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            # p_emb = self.p_embedding.astype(np.float32).toarray()
            # p_emb = self.p_embedding.toarray()
            p_emb = self.p_embedding.cpu().numpy().astype(np.float32)
            # p_emb = self.p_embedding.toarray().astype(np.float32)
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1, single_passage: Optional[bool] = True,
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

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )

            if single_passage:
                doc_scores = np.array(doc_scores)
                doc_scores = doc_scores / np.max(doc_scores)
                cqas_list = [] 
                for i in range(topk):
                    total = []
                    for idx, example in enumerate(
                        tqdm(query_or_dataset, desc="Dense retrieval: ")
                    ):
                        tmp = {
                            # Query와 해당 id를 반환합니다.
                            "question": example["question"],
                            "id": example["id"],
                            # Retrieve한 Passage의 id, context를 반환합니다.
                            "context": self.contexts[doc_indices[idx][i]],
                        }
                        if "context" in example.keys() and "answers" in example.keys():
                            # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                            tmp["original_context"] = example["context"]
                            tmp["answers"] = example["answers"]
                        total.append(tmp)
                    cqas = pd.DataFrame(total)
                    cqas_list.append(cqas)    
                return doc_scores, cqas_list
            
            else:
                for idx, example in enumerate(
                    tqdm(query_or_dataset, desc="Dense retrieval: ")
                ):
                    tmp = {
                        # Query와 해당 id를 반환합니다.
                        "question": example["question"],
                        "id": example["id"],
                        # Retrieve한 Passage의 id, context를 반환합니다.
                        "context": " ".join(
                            [self.contexts[pid] for pid in doc_indices[idx]]
                        ),
                    }
                    if "context" in example.keys() and "answers" in example.keys():
                        # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                        tmp["original_context"] = example["context"]
                        tmp["answers"] = example["answers"]
                    total.append(tmp)

                cqas = pd.DataFrame(total)
                return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        
        with timer("transform"):
            # 변경------------------------------------------------------------------------
            query_vec = self.query_embedding([query])
            # 변경------------------------------------------------------------------------
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = query_vec @ self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

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
        query_vec = self.query_embedding(queries).to('cuda')
        with torch.no_grad():
            result = query_vec @ self.p_embedding.T
            if not isinstance(result, np.ndarray):
                result = result.to('cpu').numpy()
            doc_scores = []
            doc_indices = []
            for i in range(result.shape[0]):
                sorted_result = np.argsort(result[i, :])[::-1]
                doc_scores.append(result[i, :][sorted_result].tolist()[:k])
                doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    def retrieve_faiss(
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
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.query_embedding(self, [query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        # q_emb = query_vec.toarray().astype(np.float32)
        q_emb = query_vec.toarray().cpu().numpy().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
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

        query_vecs = self.query_embedding(queries)
        # assert (
        #     np.sum(query_vecs) != 0
        # ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        # q_embs = query_vecs.toarray().astype(np.float32)
        q_embs = query_vecs.cpu().numpy().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()
    

class HybridRetrieval:
    def __init__(
            self,
            args,
            # tokenize_fn,
            p_encoder_path,
            q_encoder_path,
            data_path: Optional[str] = "../data/",
            context_path: Optional[str] = "wikipedia_documents.json"
    ):
        self.sparse_retrieval = SparseRetrieval(
            # tokenize_fn = tokenize_fn,
            args = args,
            data_path = data_path,
            context_path = context_path
            )
        self.dense_retreival = DenseRetrieval(
            args=args, 
            p_encoder_path=p_encoder_path, 
            q_encoder_path=q_encoder_path,
            data_path = data_path,
            context_path = context_path
            )
        self.q_encoder = self.dense_retreival.q_encoder
        print('Sparse Embedding Get Start')
        self.sparse_retrieval.get_sparse_embedding()
        print('Sparse Embedding Get End')

        print('Dense Embedding Get Start')
        self.dense_retreival.get_dense_embedding()
        print('Dense Embedding Get End')
        self.p_embedding = self.dense_retreival.p_embedding
        if torch.cuda.is_available():
            self.p_embedding = torch.Tensor(self.p_embedding).to("cuda")

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
            
    def retrieve(self, query_or_dataset, topk, single_passage):
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_sparse_idx_score(query_or_dataset)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])
        
        elif isinstance(query_or_dataset, Dataset):
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_sparse_idx_score_bulk(query_or_dataset["question"], topk=topk)
            
            # with timer("query exhaustive search"):
            #     doc_scores, doc_indices = self.get_relevant_doc_bulk(
            #         query_or_dataset["question"], k=topk
            #     )

                
            if single_passage:
                doc_scores = np.array(doc_scores)
                doc_scores = doc_scores / np.max(doc_scores)
                cqas_list = [] 
                for i in range(topk):
                    total = []
                    for idx, example in enumerate(
                        tqdm(query_or_dataset, desc="Hybrid retrieval: ")
                    ):
                        tmp = {
                            # Query와 해당 id를 반환합니다.
                            "question": example["question"],
                            "id": example["id"],
                            # Retrieve한 Passage의 id, context를 반환합니다.
                            "context": self.contexts[doc_indices[idx][i]],
                        }
                        if "context" in example.keys() and "answers" in example.keys():
                            # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                            tmp["original_context"] = example["context"]
                            tmp["answers"] = example["answers"]
                        total.append(tmp)
                    cqas = pd.DataFrame(total)
                    cqas_list.append(cqas)    
                return doc_scores, cqas_list


            else:
                for idx, example in enumerate(
                    tqdm(query_or_dataset, desc = "Hybrid retrieval: ")
                ):
                    tmp = {
                        "question": example["question"],
                        "id": example["id"],
                        "context": " ".join(
                            [self.contexts[pid] for pid in doc_indices[idx]]
                            ),
                    }
                    if "context" in example.keys() and "answers" in example.keys():
                        # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                        tmp["original_context"] = example["context"]
                        tmp["answers"] = example["answers"]
                    total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas


    def get_sparse_idx_score(self, query ,topk):
        spr_score, spr_idx = self.sparse_retrieval.get_relevant_doc(
            query,
            k = 100
        )
        return self.__rerank(query, spr_idx, spr_score, topk)
    
    def get_sparse_idx_score_bulk(self, queries ,topk):
        spr_score, spr_idx = self.sparse_retrieval.get_relevant_doc_bulk(
            queries,
            k = 100
        )
        return self.__rerank(queries, spr_idx, spr_score, topk)
    

    def __rerank(self, queries, spr_idx, spr_score, topk):
        final_indices = []  # 최종적으로 선택된 문서 인덱스를 저장할 리스트
        final_scores = []  # 최종 점수를 저장할 리스트

        for i, query in enumerate(queries):
            # 현재 쿼리에 대한 상위 문서 인덱스와 점수
            indices = spr_idx[i].tolist()
            scores = spr_score[i]


            # 현재 쿼리에 대한 문서 인덱스와 점수를 텐서로 변환
            indices_tensor = torch.tensor(indices, dtype=torch.long, device='cuda')
            scores_tensor = torch.tensor(scores, device='cuda')

            # 쿼리 임베딩 계산
            with torch.no_grad():
                q_emb = self.dense_retreival.query_embedding([query]).unsqueeze(0)  # 쿼리를 리스트로 변환하여 전달
                selected_p_embs = self.p_embedding[indices_tensor]

            # Dense Retrieval 점수 계산 (내적)
            dense_scores = torch.matmul(q_emb, selected_p_embs.T).squeeze()

            # 점수 결합
            combined_scores = dense_scores + scores_tensor

            # 상위 k개 문서 선택
            _, topk_indices = torch.topk(combined_scores, k=topk)

            # 선택된 문서 인덱스와 점수 저장
            selected_indices = indices_tensor[topk_indices].cpu().tolist()
            selected_scores = combined_scores[topk_indices].cpu().tolist()

            final_indices.append(selected_indices)
            final_scores.append(selected_scores)

        return final_scores, final_indices
    
