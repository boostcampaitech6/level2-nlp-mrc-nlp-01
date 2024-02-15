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


train.py --model_name=klue/bert-base --output_dir ../parameters --batch_size 32 --do_train
```


### Inference

```
python3 inference.py --output_dir ../output --dataset_name ../data/test_dataset/ --model_name_or_path ../parameters/kcbert-base-finetuned-squad --do_predict --overwrite_output_dir True
```




### Results

|model name| batch size | epochs | Extract Macth | F1 Score | 
|:----:|:----:|:----:|:----:|:----:|
|klue/bert-base| 32 | 3 | **38.3300** | **49.0300** |
|klue/bert-base | 32 | 10 | 38.3300 | 49.0300 | 
|ainize/klue-bert-base-mrc| 32 | 3 | 37.5000 | 48.5800 |
|kcbert-base-finetuned-squad | 32 | 10 | 22.9200 | 35.8100 | 
|Forturne/bert-base-finetuned-klue-mrc | 32 | 10 |36.6700 | 49.2100 |
|domyoung/squad-test | 32 | 10 |41.2500 | 51.8100 |
|Seonwhee-Genome/bert-base | 32 | 10 |40.0000 | 51.7800 |
