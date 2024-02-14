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





### Results

|model name| batch size | epochs | Extract Macth | F1 Score | 
|:----:|:----:|:----:|:----:|:----:|
|klue/bert-base| 32 | 3 | **38.3300** | **49.0300** |
|klue/bert-base | 32 | 10 | | | 
|ainize/klue-bert-base-mrc| 32 | 3 | 37.5000 | 48.5800 |
|kcbert-base-finetuned-squad | 32 | 10 |  | 
