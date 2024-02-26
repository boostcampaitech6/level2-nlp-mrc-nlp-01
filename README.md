## Machine Reading Comprehension

### Dataset 
```
from datasets import Dataset

path = 'dataset.arrow'
data = Dataset.from_file(path)
data
# Dataset({
#     features: ['title', 'context', 'question', 'id', 'answers', 'document_id', '__index_level_0__'],
#     num_rows: 3952
# })
```


### Training
```
# in arguments.py
class ModelArguments:
    ...
    batch_size:int = field(
    default=16
    )


# in train.py
def main():
    ....
    training_args.per_device_train_batch_size = model_args.batch_size


train.py --model_name_or_path klue/bert-base --output_dir ../parameters/klue-bert-base --batch_size 32 --num_epochs 10 --overwrite_output_dir True --do_train
```


### Inference
```
python3 inference.py --output_dir ../output --dataset_name ../data/test_dataset/ --model_name_or_path ../parameters/kcbert-base-finetuned-squad --do_predict --overwrite_output_dir True
```

# 1. 프로젝트 개요
### 1.1 프로젝트 주제
![image](https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-01/assets/88371786/269f5fa4-9e08-4283-88e4-4d5f4b4c62a0)

- **Question Answering (이하 QA)** 은 다양한 종류의 질문에 대답하는 인공지능을 만드는 태스크
- **Open-Domain Question Answering (이하 ODQA)** 은 **주어지는 지문 없이 사전에 구축된 Knowledge resource에서** 질문에 대답할 수 있는 문서를 앞 부분에 추가한 태스크
- 본 ODQA 대회에서 만들 모델은 두 단계로 구성
    - 첫 단계는 질문에 관련된 **Knowledge resource 내 문서를 찾아주는 "retriever"** 단계
    - 다음으로는 관련된 문서를 읽고 **적절한 답변을 찾거나 만들어주는 "reader"** 단계

### 1.2 프로젝트 구현 내용
- 본 프로젝트에선 Retrieval과 Reader 부분을 여러 실험을 통해 개선하여, 최종 Score를 높이는 것이 목표.
- 두 가지 sub-task가 혼합된 프로젝트, **두 document 간 유사도를 높여야 하는 retrieval, answer span을 학습하는 reader** 각각의 성능 향상 중요.

### 1.3 활용 장비 및 재료(개발 환경, 협업 tool 등)
- VS Code + SSH 접속을 통해 AI stage 서버 GPU 활용
- Git을 통한 버전 관리, Github를 통한 코드 공유 및 모듈화
- Slack, Zoom, Notion을 통한 프로젝트 일정 및 문제 상황 공유 + 회의 진행

### 1.4 데이터셋 설명
- 기본제공 데이터셋 train.dataset, test.dataset이 주어짐
- train_datasets은 ‘id’, ‘question’ , ‘context’ , ‘answer’ , ‘document_id’ , ‘title’ 총 6개의 columns으로 총 4000여개, test_datasets은 ‘id’ , ‘question’ 총 2개의 columns에 600개
- Retrieval 과정에서 사용하는 문서 집합(corpus)은 위키백과에 등재된 내용으로, 5만 7천개의 문서로 이루어진 json

### 1.5 평가 지표
- EM, F1 두 가지 평가지표.
- Exact Match(EM): 모델의 예측과, 실제 답이 정확하게 일치할 때만 점수가 주어진다. 즉 모든 질문은 0점 아니면 1점으로 처리된다.
- F1 Score: EM과 다르게 부분 점수를 제공한다. 예를 들어, 정답은 "Barack Obama"지만 예측이 "Obama"일 때, EM의 경우 0점을 받겠지만, F1 Score는 겹치는 단어도 있는 것을 고려해 부분 점수를 받을 수 있다.

# 2. 프로젝트 팀 구성 및 역할
- 김인수(팀장) : Retrieval(BM25 Sparse, DPR, Hybrid) 구현, 모듈화
- 김동언(팀원) : Reader(모델 실험, Hyper-parameter Tuning)
- 오수종(팀원) : Question-generation(KorQuAD 1.0 fine-tuning), Ensemble
- 이재형(팀원) : Hard Negative Sample(Sparse Retrieval) 탐색, 특수문자 제거 전처리 구현, KorQuAD 2.0 transfer learning
- 임은형(팀원) : Retrieval(BM25, BM25+TF-IDF) 구현, Back-translation, Hyper-parameter Tuning
- 이건하(팀원) : Hard Negative Sampling(Dense Retrieval) 탐색, Single Passage

# 3. 프로젝트 수행 절차 및 방법
### 3.1 팀 목표 설정
![image](https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-01/assets/88371786/b3b6fa4b-0819-4e2f-9f12-c4c2f3936092)

### 3.2 프로젝트 사전기획
- Notion : 조사 및 실험 결과 공유, 기록 정리 및 보관
- Github: 공용 코드 정리, 모듈화 및 코드리뷰, 진행사항 공유
- Zoom : 화상 회의를 통한 실시간 상황 공유, 결과 피드백, 이슈 해결

### 3.3 프로젝트 수행
![image](https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-01/assets/88371786/1012e339-b1a4-43f1-bd8d-902bf66f2f3b)

| Retrieval | Reader | Inference | EDA |
| --- | --- | --- | --- |
| 1. Sparse: TF-IDF, BM25, BM25+TF-IDF | 1. Pre & Post Processing: tokenizer | 1. Hyper-parameter Tuning: top-k, doc_stride | 1. Hard Negative	Sampling EDA | 
| 2. Dense: DPR | 2. Transfer learning: korquad-2.0 | 2. Ensemble(soft voting)| 2. 특수기호 영향 파악 |
| 3. Hybrid: Sparse+Dense | 3. Data_augmentation: korquad-1.0, back-translation| 
| 4. Pre-processing: Remove Special Char | 4. Single passage prediction | 

# 4. 프로젝트 수행 결과
## 4.1 EDA
### **4.1.1 Retrieval Hard Negative Sampling**
- **Sparse(Check Negative Sample)**
    ![image](https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-01/assets/88371786/433f5d9a-9bd4-413c-9190-e2b199a70b88)
    
    - Sparse Retrieval 후 Hard-negative Sample에 대한 정성적 분석 진행, 즉 점수는 높지만, 실제 ground-truth passage와 다른 값들 위주로 분석
    - 더 유사한 pair가 있는 경우도 있지만, “/n”이 “\n”으로 텍스트 내에 생성되는 이슈 발견 →  Retrieval matching 이전에 passage에서 특수문자 제거하는 방법 구상
    - 사람이 헷갈릴만한 Negative Sample도 많았음
        - 질문: 우라노스가 가이아의 자식들을 가둔 곳은?
        - 반환된 답변: 가이아가 스스로 낳은 첫 번째 자식으로, 이후 가이아가 장남인 우라노스를 남편으로 …
        - Ground-Truth: 우라노스는 아내 가이아가 낳은 자식들 헤카톤케이레스, 키클롭스, 티탄들을 보기 싫다...

### **4.1.2 Annotation bias**
- QA 데이터셋에서 question 생성 시, passage의 내용에 dependent한 **annotation bias** 발견
- 이로 인해 실제 질문이 passage의 내용으로 구성되는 경우가 많아, **sparse가 성능이 좋고** dense retrieval 성능에 한계
    ![image](https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-01/assets/88371786/a556f19b-5bd3-40d9-bff5-b15f0643a2f4)

### **4.1.3 Data Length**
- Train/Validation Datasets의 Answer Length 확인 결과, 전체적으로 40 character를 넘어가지 않는 것을 확인
- Reader의 하이퍼파라미터 **max_answer_length를 64로 고정**하여 프로젝트 진행
![image](https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-01/assets/88371786/8ba3076b-58c4-460c-b749-5e03eddc3afa)
![image](https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-01/assets/88371786/222228a6-4362-425f-9373-96c2dced22a6)


## 4.2 Retrieval
### 4.2.1 Sparse Retrieval
- TF-IDF: 단어의 문서 내 등장 빈도가 높으면서(TF), 다른 문서에서는 등장하지 않는(IDF) 단어에 가중치를 부여하는 기법
    ![image](https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-01/assets/88371786/f1a7fd22-a93e-4ba5-8f3f-65a0cc489b11)
    
- BM25: TF-IDF의 개념을 바탕으로, 문서의 길이까지 고려하는 기법
    ![image](https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-01/assets/88371786/4d5acba3-cc98-41c4-bee8-7c6551dbfd92)
    
- TF-IDF + BM25 : TF-IDF 점수 와 BM25 점수를 각각 정규화 한 후 더하여 최종 점수로 사용 
→ TF-IDF와 BM25를 사용하여 예측한 결과를 직접 비교한 결과, 두 방법 간에는 각기 다른 장점이 있었고 이에 따라 두 방법을 함께 활용하면 더 효과적인 예측이 가능할 것으로 판단되어, 두 기법을 결합하여 사용
- Tokenizer : kiwi, soynlp, mecab 형태소 분석기 사용 결과 mecab 성능이 가장 우수, mecab과 roberta tokenizer 사용
- 결과: TF-IDF(EM:42.5000, F1:53.9200), BM25(EM:47.9200 , F1:59.0500)를 개별적으로 적용하는 것보다, TF-IDF+BM25 방법(EM:55.0000, F1:66.3600)의 성능이 가장 우수

### 4.2.2 Dense Retrieval
- Sparse Embedding의 단점인 단어의 유사성 또는 맥락을 파악하지 못하는 경우를 보완하기 위해, Dense Embedding을 통한 **Dense Passage Retrieval(DPR)** 논문 구현 [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2004.04906.pdf)
- question(q), passage(p) Encoder KorQuAD 1.0 데이터셋으로 **in-batch negative sampling** 방법을 사용해서 fine-tune.(inner product 값이 높은 q,p 쌍은 벡터 공간에서 거리를 가깝게(positive), 낮은 경우 거리를 멀게(negative) 만드는 방식.)
- negative example 선택 방법은 논문에서 다룬 BM25 유사도 점수가 높지만 answer를 포함하지 않는 경우를 negative example로 선택하는 방식 사용. → KorQuAD dataset의 각 question 마다 BM25 유사도 점수가 높은 top 5 context에서 랜덤 선택하여 negative sample로 추가하여 데이터셋 변형.
    
    ![image](https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-01/assets/88371786/32be831e-9861-4690-8371-6897bf4ba00c)
- 결과: EDA 결과 발견된 데이터셋 내 annotation bias로 인해 단어의 overlap을 효과적으로 잡을 수 있는 Sparse Retrieval이 더 좋은 성능을 내는 모습

### 4.2.3 Hybrid Retrieval (Sparse+Dense)
- Sparse Retrieval의 단어의 overlap을 잘 잡을 수 있는 장점과 Dense Retrieval의 단어의 유사성 또는 맥락을 파악할 수 있는 장점을 합칠 수 있는 **Hybrid Retrieval** 구현
- Sparse Retrieval(BM25)를 통해 question에 대한 상위 100개의 passage의 similarity를 계산하고, 해당 pssage들에 대해 Dense Retrieval를 추가적으로 수행하여 Sparse Retrieval similarity와 Dense Retrieval inner product score를 더하여 rerank한 후 topk의 passage를 reader에 전달.
- 결과: EM:54.1700, F1: 67.3900으로 BM25 기반 Sparse Retrieval보다 높은 성능.

### 4.2.4 Pre-processing
- Special Character로 인해 Embedding 값에 노이즈가 섞인다고 가정
- 전처리 과정을 통해 passage를 임베딩하기 전에, Special Character 제거하는 과정 거침
- 결과: EM: 42.9200, F1: 58.6600으로 TF-IDF 기반 Sparse Retrieval 보다 성능 향상.

## 4.3 Reader
### 4.3.1 Transfer learning: korquad-2.0
- 삼성SDS에서 진행한 Techtonic 2020. “(KorQuAD 1.0성능개선 Know-how)”의 발표에 근거하여, KorQuAD Dataset을 이용해 Pre-training을 먼저 진행한 후, 대회에서 사용된 데이터셋을 이용해 Fine-tuning 진행
- 결과: EM 52.5000, F1 65.6500으로 기존 KLUE-MRC 데이터 기반 모델 보다 성능 향상

### 4.3.2 Data_augmentation
- Question-generation
    - 먼저 KorQuAD 1.0 dataset으로 context, question,answers을 모델에 학습 시킨 후, wiki data에서 title이 text안에 있는 경우만 추출하여 datasets을 생성.
    - text 에서 title이 정답이 되도록 하는 question을 생성하여 train_data에 추가하여 학습
    - 결과: EM 27.0800, F1 36.0900으로 기존 데이터를 사용했을 때 보다 성능 하락
    - (generation된 question을 보면 잘 된 경우도 있지만 , 잘되지 않은 경우가 많이 존재 이런 노이즈로 인한 성능저하 추정)

### 4.3.3 back-translation
- mbart-large 모델을 사용하여 test data의 question을 (iterative)back-translation을 통해 데이터 증강. (T5모델 보다는 mbart-large 모델이, iterative하게 진행한 방식이 더 나은 결과를 보임.)

### 4.3.4 Single passage
- Multi-passage(Baseline 코드): k개의 document를 concat한 뒤 그 안에서 answer span을 찾는 프로세스(전체 document에 대한 answer span 확률이 할당)
- (Single passage): document를 concat하지 않고, 각 document에 대해 answer span의 확률값을 예측할 경우, 예측 결과가 달라질 수 있다는 가정 하에 reader가 document를 하나씩 보고(각 document에 대한 answer span확률이 할당) 최종 answer span에 대한 확률을 출력하도록 설정
- Probability로 비교할 경우 document의 answer span 개수의 영향을 받을 것을 우려하여, logit값을 통한 비교 진행
- BM25를 적용한 Sparse Retrieval에 대해 EM:47.9200/F1:59.0500 → EM:55.8300/F1:69.1500로 점수가 크게 향상되는 효과

### **4.3.5 Post Processing**
- 정답 내에 「 」, << >> , 『 』, « », “ ” , ‘ ’와 같은 특수 문자가 포함될 경우 제거
- answer span이 end span보다 앞 token의 index를 가리킬 경우 정답에서 제외
- answer text의 길이가 0이거나, max_answer_length보다 큰 경우 정답에서 제외

## 4.4 Hyper-parameter Tuning & Ensemble
### 4.4.1 Hyper-parameter Tuning
- top-k: **retriever의 결과로 몇 개의 상위 Document**를 가져와 reader에서 사용할 지를 결정하는 하이퍼파라미터. 10 → 20으로 변경하여 성능
- doc_stride: 긴 document를 chunk할 때, **stride를 얼마나 줄지**에 대한 하이퍼 파라미터. 64, 256 등으로 실험 결과 기본값인 128 그대로 사용
- Optimizer : AdamW → AdamP
- Scheduler : get_linear_schedule_with_warmup → CosineAnnealingLR

### 4.4.2 Ensemble
- soft voting: nbest의 각 문제 당 예측 답에 대한 probability를 앙상블을 할 모델들 전체에 합쳐서 그 중 가장 probability를 높은 것으로 선정
- sparse가 가지는 장점과, dense의 강점이 달라, Dense의 단일 모델의 성능은 낮지만 같이 앙상블을 적용

| Model | Retrieval | Tokenizer | 특이사항 |
| --- | --- | --- | --- |
| klue/roberta-large | Hybrid | roberta-large | mecab |
| klue/roberta-large | Hybrid | roberta-large | dense 0.8 |
| uomnf97/klue-roberta-finetuned-korquad-v2 | BM25plus+TF-IDF | uomnf97/klue-roberta-finetuned-korquad-v2 | 특수 문자 제거 |
| uomnf97/klue-roberta-finetuned-korquad-v2 | Hybrid | roberta-large | single passage, mecab |

## 4.5 Final Submission
- Public LeaderBoard
    ![image](https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-01/assets/88371786/808dd65b-c41a-4f5d-8173-3484e9d740da)

- Private LeaderBoard
    ![image](https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-01/assets/88371786/8e844614-6983-4ffa-ae42-d0462680fd9f)

# 5. 자체 평가 의견
### 5.1 잘한 점
- Github, Zoom 등 협업이 원활하게 이루어져서 실험 단계에서 다양한 요소를 실험해볼 수 있었음
- Hybrid Retrieval, Transfer learning 등 다소 어려운 방법론을 적극적으로 구현하고 시도해 봄

### 5.2 시도 했으나 잘 되지 않았던 것
- Reader 파트에서 모델 구조를 변경하여 성능 향상 시도를 못해 본 점.

### 5.3 아쉬웠던 점 → 개선 방안
- 실험에 대한 기록이 초반에 이뤄지지 않아서 팀원 간 의사소통에 병목이 있었음 → 이전 대회처럼 진행상황에 대한 기록을 적극적으로 하기
- baseline 코드가 복잡해서 팀원들이 이해하기까지 어려움이 있었음 → 다소 공수가 있더라도 팀원들이 bottom-up으로 작은 규모의 코드를 직접 구성하기
- 복잡한 baseline 코드를 refactoring 하지 않아 팀원들간 코드 충돌이 잦아 Github 공유가 활발하지 못한 점 
→ 복잡한 구조의 코드를 보기 쉽고 명확하게 재구성하기.
- 데이터 증강을 시도하였지만 성능 하락으로 이어짐 → 모델이 잘 맞추지 못하는 부분을 중심으로 데이터를 증강해서 사용했으면 좋았을 것 같음.(시간 부족으로 시도하지 못함)
- Retrieval Hard Negative Sampling을 위해, Ground Truth가 아니지만 Query와 임베딩 공간상 비슷한 특성을 갖는 context를 Negative Sample로 지정해서 훈련하기 위해 clustering을 진행했으나(CLS Embedding), 의미있는 cluster를 형성하지 못하여 사용하지 않음
→ (Dense 유사도를 사용하지 않고, Sparse 유사도가 높은 sample만을 이용해 in-batch Negative Sampling 진행)
→ Cluster를 꼭 나누기보다, Pruning - Inverted File(IVF)처럼, 가까운 N개에 대해 Dense Negative Sampling도 진행해봤으면 하는 아쉬움
    ![image](https://github.com/boostcampaitech6/level2-nlp-mrc-nlp-01/assets/88371786/d9beb2a0-a5b4-4014-a653-b52ebfc8fa01)
    
### 5.4 프로젝트를 통해 배운 점 또는 시사점
- 막연하고 어렵게만 느껴졌던 ODQA 같은 복잡한 태스크를 서로 작은 sub-task로 분할한 뒤 차근차근 해결할 수 있다는 점을 배움
- 프로젝트를 시작할 때에는 Dense Passage Retrieval이 Sparse Retrieval보다 우수한 성능을 보일 것으로 예상했지만, 실제 프로젝트에서는 Sparse Retrieval이 더 효과적이었음. 데이터 특성 상 검색어가 대부분의 Passage에 존재하는 경우가 많아서, 이번 태스크에서는 단어의 의미와 문맥을 고려하는 DPR보다는 검색어와 정확하게 일치하는 문서를 찾는 데 탁월한 성능을 보이는 Sparse Retrieval을 사용하였을 때 성능 개선이 있었다고 생각함. 이를 통해 최신 연구와 모델을 적용하는 것 뿐만 아니라 데이터에 적합한 방법을 찾는 것 또한 중요하다는 것을 느낌.

## Reference
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2004.04906.pdf)
- https://huggingface.co/uomnf97/klue-roberta-finetuned-korquad-v2
|domyoung/squad-test | 32 | 10 |41.2500 | 51.8100 |
|Seonwhee-Genome/bert-base | 32 | 10 |40.0000 | 51.7800 |
